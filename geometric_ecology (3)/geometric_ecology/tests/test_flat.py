"""Numerical controls for true-2D isolines, masking and field independence."""
import unittest
import numpy as np
from geometric_ecology.flat import field_contours, enrich_flat
from geometric_ecology.core import build_map, normalize
from geometric_ecology.atlas import build_atlas
from geometric_ecology.terrain import enrich_terrain
from geometric_ecology.mesh import enrich_mesh


class FlatTests(unittest.TestCase):
    def plane(self):
        n=11
        return dict(size=n,bounds=[0,0,10,10],valid=np.ones((n,n),int),
                    values={'value':np.tile(np.arange(n,dtype=float),(n,1))})

    def test_contours_are_actual_field_values(self):
        field=self.plane()
        paths=field_contours(field,'value',[2.5,7.25])
        for record in paths:
            self.assertGreater(len(record['paths']),0)
            for path in record['paths']:
                np.testing.assert_allclose(np.asarray(path)[:,0],record['level'],atol=1e-5)

    def test_masked_gap_breaks_lines_instead_of_bridging(self):
        field=self.plane();field['valid'][4:7,:]=0
        record=field_contours(field,'value',[5.5])[0]
        self.assertEqual(len(record['paths']),2)
        for path in record['paths']:
            ys=np.asarray(path)[:,1]
            self.assertTrue(np.all(ys<=3) or np.all(ys>=7))

    def test_constant_and_invalid_fields_gain_no_artificial_contours(self):
        field=self.plane();field['values']['value'][:]=4
        self.assertTrue(all(not r['paths'] for r in field_contours(field,'value',[1,3,5,8])))
        field['values']['hidden_two']=np.tile(np.linspace(0,1,11),(11,1))
        field['values']['covariance_valid']=np.zeros((11,11))
        self.assertFalse(field_contours(field,'hidden_two',[.5])[0]['paths'])

    def test_flat_fields_do_not_change_atlas_positions_or_connections(self):
        x=normalize(np.random.default_rng(12).normal(size=(200,12)))
        model=build_map(x,scales=[6,12,24],regions=12,repeats=2)
        atlas=build_atlas(model);atlas['terrain']=enrich_terrain(model,atlas)
        atlas['mesh']=enrich_mesh(model,atlas)
        xy=np.asarray(atlas['xy']).copy();edges=model['stable_edges'].copy()
        flat=enrich_flat(model,atlas)
        np.testing.assert_array_equal(xy,atlas['xy'])
        np.testing.assert_array_equal(edges,model['stable_edges'])
        self.assertEqual(len(flat['fields']),9)
        for island in atlas['islands']:
            f=island['field'];b=f['bounds']
            for definition in flat['fields']:
                key=definition['key']
                self.assertTrue(np.isfinite(f['values'][key]).all())
                for record in f['isolines'][key]:
                    for path in record['paths']:
                        points=np.asarray(path)
                        self.assertTrue(np.all(points[:,0]>=b[0]-1e-4))
                        self.assertTrue(np.all(points[:,0]<=b[2]+1e-4))
                        self.assertTrue(np.all(points[:,1]>=b[1]-1e-4))
                        self.assertTrue(np.all(points[:,1]<=b[3]+1e-4))


if __name__=='__main__':unittest.main()
