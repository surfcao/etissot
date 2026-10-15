"""Paired fixed-weight pi-correction experiment and cached manuscript replay."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from etissot.geometry.fitting import fit_many
from etissot.geometry.probing import probe_points
from etissot.statistics.spatial import wquantile

BEARINGS = np.arange(32)*2*np.pi/32


def collect(workspace, model_dir):
    import torch
    from etissot.models.satclip_correction import correct_zonal_normalization
    vendor = Path(__file__).resolve().parents[1]/'models/vendor/satclip/satclip'
    from etissot.models.loaders import load_satclip_checkpoint
    torch.set_num_threads(2)
    sites = np.array([(lat,lon) for lat in [-75,-50,-25,0,25,50,75]
                      for lon in [-150,-75,0,75,150]], dtype=float)
    radii = np.array([1., .5])
    coordinates = []
    for radius in radii:
        for lat,lon in sites:
            la,lo,_ = probe_points(*np.radians([lat,lon]), radius)
            coordinates.append(np.degrees(np.stack([lo,la],axis=1)))
    x = torch.tensor(np.concatenate(coordinates), dtype=torch.float64)
    expected = json.loads((workspace.data/'reference/summaries/checkpoint_manifest.json').read_text())
    expected_hashes = {Path(v['checkpoint']).name:v['sha256'] for v in expected.values()}
    profiles = []; input_profiles = []; metadata = []
    for backbone in ['vit16','resnet18','resnet50']:
        for degree in [10,40]:
            paths = list(Path(model_dir).rglob(f'satclip-{backbone}-l{degree}.ckpt'))
            if len(paths)!=1:raise ValueError(f'Expected one checkpoint: {backbone}, {degree}')
            path = paths[0]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected_hashes.get(path.name) != digest:
                raise ValueError(f'Checkpoint does not match the pinned release: {path.name}')
            # These are the pinned trusted local upstream checkpoints.
            import os
            model = load_satclip_checkpoint(path,'cpu').double().eval()
            corrected = correct_zonal_normalization(model)
            compensated = correct_zonal_normalization(model,preserve_outputs=True)
            with torch.no_grad():
                original_input = model.posenc(x)
                scale = original_input.new_ones(degree**2)
                scale[[l*(l+1) for l in range(degree)]]=1/np.pi
                corrected_input = original_input*scale
                outputs = [model.nnet(original_input), corrected.nnet(corrected_input),
                           compensated.nnet(corrected_input)]
                outputs = [torch.nn.functional.normalize(y,dim=1).numpy() for y in outputs]
            def profile(y):
                y = y.reshape(len(radii),len(sites),33,-1)
                return np.linalg.norm(y[:,:,1:]-y[:,:,:1],axis=-1)/radii[:,None,None]
            profiles.append(np.stack([profile(y) for y in outputs]))
            input_profiles.append(np.stack([profile(original_input.numpy()),profile(corrected_input.numpy())]))
            centers = [y.reshape(len(radii),len(sites),33,-1)[0,:,0,:] for y in outputs]
            meta = dict(name=f'{backbone}-L{degree}',degree=degree,
                        checkpoint=path.name,sha256=digest,
                        compensated_max_output_difference=float(np.max(abs(outputs[0]-outputs[2]))),
                        corrected_center_cosine=np.sum(centers[0]*centers[1],axis=1).tolist())
            metadata.append(meta)
            print(meta['name'], 'measured',flush=True)
    destination=workspace.output/'profiles/satclip_correction.npz'
    np.savez_compressed(destination, sites=sites,radii=radii,output_profiles=np.stack(profiles),
                        input_profiles=np.stack(input_profiles),metadata=json.dumps(metadata),
                        output_variants=np.array(['released','pi_corrected_fixed_weights','pi_corrected_compensated']),
                        input_variants=np.array(['released','pi_corrected']),
                        protocol='float64, 32 bearings, unit-normalised outputs; pi-only correction; original expanded harmonic basis retained')
    return destination


def run(workspace):
    source=workspace.data/'profiles/satclip_correction.npz'
    with np.load(source,allow_pickle=False) as d:
        metadata=json.loads(str(d['metadata']));sites=d['sites'];radii=d['radii'];weights=np.cos(np.radians(sites[:,0]))
        rows=[]
        for i,meta in enumerate(metadata):
            row=dict(meta);row['radii']={}
            for j,radius in enumerate(radii):
                results={}
                for group,key,names in [('output','output_profiles',['released','corrected','compensated']),
                                         ('input','input_profiles',['released','corrected'])]:
                    for k,name in enumerate(names):
                        fit=fit_many(d[key][i,k,j],BEARINGS)
                        results[group+'_'+name]=dict(median_A=float(wquantile(fit['A_fit'],.5,weights)),
                            max_A=float(fit['A_fit'].max()),fallbacks=int(fit['fallback'].sum()),
                            A=fit['A_fit'].tolist())
                released=np.array(results['output_released']['A']);comp=np.array(results['output_compensated']['A'])
                results['compensated_max_relative_A_difference']=float(np.max(abs(comp/released-1)))
                row['radii'][str(float(radius))]=results
            row['median_corrected_center_cosine']=float(wquantile(meta['corrected_center_cosine'],.5,weights))
            rows.append(row)
    report=dict(scope='Selected-site fixed-weight input intervention; not retraining or downstream accuracy validation.',
                sites=sites.tolist(),radii_km=radii.tolist(),bearings=32,weighting='cos(latitude)',
                basis='pi-only correction; high-degree expanded-polynomial errors remain',checkpoints=rows)
    (workspace.output/'summaries/satclip_correction.json').write_text(json.dumps(report,indent=2)+'\n')
    text=[r'\begin{table}[htbp]\centering\small',
          r'\caption{Paired SatCLIP input rescaling at 35 selected sites and 1\,km radius. Entries are area-weighted medians of fitted anisotropy; cosine is the area-weighted median cosine similarity between released and rescaled unit outputs at the site centers.}',
          r'\label{tab:satclipcorrection}',
          r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
          r'&\multicolumn{2}{c}{Input $A$}&\multicolumn{2}{c}{Output $A$}&\\',
          r'Checkpoint&Released&Rescaled&Released&Rescaled&Cosine\\\midrule']
    for row in rows:
        r=row['radii']['1.0']
        text.append(row['name'].replace('vit16','ViT').replace('resnet18','R18').replace('resnet50','R50')+'&'+'&'.join(f"{r[k]['median_A']:.3f}" for k in ['input_released','input_corrected','output_released','output_corrected'])+f"&{row['median_corrected_center_cosine']:.3f}"+r'\\')
    text += [r'\bottomrule\end{tabular}',
             r'\par\smallskip\parbox{.97\textwidth}{\footnotesize Medians include flagged fallback values. Released/rescaled fallback counts at 1 km, in row order: '+', '.join(str(row['radii']['1.0']['output_released']['fallbacks'])+'/'+str(row['radii']['1.0']['output_corrected']['fallbacks']) for row in rows)+r'.}',
             r'\end{table}']
    maximum_input = max(row['radii']['1.0']['input_corrected']['max_A'] for row in rows)
    maximum_half = max(row['radii']['0.5']['input_corrected']['max_A'] for row in rows)
    preservation = max(row['compensated_max_output_difference'] for row in rows)
    mantissa, exponent = f'{preservation:.1e}'.split('e')
    preservation_tex = mantissa + r'\times10^{' + str(int(exponent)) + '}'
    text += [f'The rescaled L10 input has fitted anisotropy 1 to the displayed precision. The rescaled L40 input has a maximum of $A={maximum_input:.5f}$ at 1 km and $A={maximum_half:.5f}$ at 0.5 km. The value is larger at the smaller radius, as expected for evaluation error (Section~\\ref{{supp:geometry}}), so dividing latitude-only channels by $\\pi$ does not remove the expanded-polynomial numerical errors. Output anisotropy remains above one, with no consistent reduction across the L40 checkpoints. These tests assess geometry and numerical equivalence, not predictive performance.']
    (workspace.output/'tables/satclip_correction.tex').write_text('\n'.join(text)+'\n')
    return report
