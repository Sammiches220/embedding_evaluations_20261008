"""Independent measured fields and masked isolines for the flat atlas."""
import time
import contourpy
import numpy as np
from .terrain import sample_fields


def field_definitions():
    """Shared scales across components; levels are values, not plot heights."""
    return [
        dict(key='hidden_two',name='Variation hidden by two axes',unit='fraction',
             minimum=0,maximum=1,levels=[.1,.2,.3,.4,.5,.6,.7,.8,.9],palette='dimension',
             description='Fraction of original-neighborhood covariance outside its first two axes. Teal retains most variation; purple hides much more.'),
        dict(key='covariance_dimensions',name='Effective local directions',unit='directions',
             minimum=1,maximum=20,levels=[1.5,2,2.5,3,4,5,6,7,8,10,12,16,20],palette='dimension',
             description='Covariance participation ratio. Contours labelled 2, 4 or 8 enclose those measured direction counts; it is not an integer rank.'),
        dict(key='density',name='Original-space density rank',unit='percentile',
             minimum=0,maximum=1,levels=[.1,.25,.5,.75,.9],palette='density',
             description='Inverse neighbor-radius percentile in the original space. This is independent of how crowded the display looks.'),
        dict(key='substitutes',name='Near-substitute fraction',unit='fraction',
             minimum=0,maximum=1,levels=[.05,.1,.25,.5,.75,.9],palette='density',
             description='Share of signature neighbors within the declared chord tolerance. A dense plain also needs low dimension and redundancy.'),
        dict(key='route_count',name='Regional route alternatives',unit='routes',
             minimum=1,maximum=12,levels=[1.5,2,3,4,6,8,10,11.5],palette='routes',
             description='Median effective count of up to 12 near-shortest regional routes to reference destinations. Bundles identify their actual destination.'),
        dict(key='resampling_loss',name='Reference-subset disagreement',unit='fraction',
             minimum=0,maximum=1,levels=[.1,.25,.5,.75,.9],palette='stability',
             description='One minus conditional neighbor overlap under reference resampling. Missing references are handled by the backend comparison.'),
        dict(key='compression_loss',name='Compression disagreement',unit='fraction',
             minimum=0,maximum=1,levels=[.1,.25,.5,.75,.9],palette='stability',
             description='One minus neighbor overlap under the recorded moderate PCA compression. It can reflect discarded structure or noise.'),
        dict(key='variation',name='Metric-variation rank',unit='percentile',
             minimum=0,maximum=1,levels=[.1,.25,.5,.75,.9],palette='variation',
             description='Percentile of local neighbor-distance coefficient of variation. This is a separate measure from dimension.'),
        dict(key='fog',name='Combined geometry disagreement',unit='fraction',
             minimum=0,maximum=1,levels=[.1,.25,.5,.75,.9],palette='stability',
             description='Maximum reference-subset and moderate-compression disagreement. Hatching can display it independently of the color field.')]


def field_contours(field, key, levels):
    """Trace isolines only inside quads with four supported corner samples."""
    values=np.asarray(field['values'][key],dtype=float)
    valid=np.asarray(field['valid'],dtype=bool)&np.isfinite(values)
    if key in ('hidden_two','covariance_dimensions') and 'covariance_valid' in field['values']:
        valid &= np.asarray(field['values']['covariance_valid'])>.5
    b=field['bounds'];n=field['size']
    generator=contourpy.contour_generator(x=np.linspace(b[0],b[2],n),
        y=np.linspace(b[1],b[3],n),z=np.ma.array(values,mask=~valid),
        name='serial',corner_mask=False,line_type='Separate')
    return [dict(level=float(level),paths=[np.round(path,5).tolist() for path in generator.lines(level)
                                         if len(path)>=2]) for level in levels]


def enrich_flat(model,atlas):
    start=time.perf_counter();xy=np.asarray(atlas['xy']);t=atlas['terrain']
    definitions=field_definitions()
    dimension=next(d for d in definitions if d['key']=='covariance_dimensions')
    dimension['maximum']=float(atlas['mesh']['covariance_cap'])
    dimension['levels']=[v for v in dimension['levels'] if v<=dimension['maximum']]
    for island in atlas['islands']:
        inds=np.flatnonzero(model['component']==island['component'])
        extras=sample_fields(xy[inds],dict(
            covariance_valid=np.asarray(atlas['mesh']['valid'],dtype=float)[inds],
            resampling_loss=np.asarray(t['resampling_loss'])[inds],
            compression_loss=np.asarray(t['compression_loss'])[inds],
            route_count=np.asarray(t['route_count'])[inds]),model['region'][inds])
        island['field']['values'].update(extras['values'])
        island['field']['isolines']={d['key']:field_contours(island['field'],d['key'],d['levels'])
                                    for d in definitions}
    return dict(schema_version=1,fields=definitions,elapsed_s=time.perf_counter()-start,
        note='Color and contour channels are independent. All positions remain strictly 2D; masked fields and contours never join disconnected component charts.')
