"""Controls for the dimensionality fields and measured local meshes."""
import unittest
import numpy as np
from geometric_ecology.core import normalize, knn, build_map
from geometric_ecology.atlas import build_atlas
from geometric_ecology.terrain import local_lens, enrich_terrain
from geometric_ecology.mesh import neighborhood_spectra, neighborhood_hull, enrich_mesh
from geometric_ecology.terrain_lab import vector_fixture


class MeshTests(unittest.TestCase):
    def test_batched_spectrum_matches_independent_local_svd(self):
        x=normalize(np.random.default_rng(5).normal(size=(90,12)))
        _,ids=knn(x,16)
        result=neighborhood_spectra(x,ids,batch_size=19)
        for index in [0,18,19,89]:
            lens=local_lens(x,index,ids[index])
            expected=np.pad(lens['spectrum'],(0,16-len(lens['spectrum'])))
            np.testing.assert_allclose(result['spectrum'][index],expected,atol=1e-6)
            self.assertAlmostEqual(result['effective'][index],lens['participation_ratio'],places=4)
            self.assertAlmostEqual(result['hidden_two'][index],1-lens['visible_two_fraction'],places=5)

    def test_measured_hull_keeps_flat_plane_and_volume_distinct(self):
        for d,rank in [(2,2),(12,3)]:
            lens=vector_fixture(d);hull=neighborhood_hull(lens)
            self.assertEqual(hull['rank'],rank)
            np.testing.assert_array_equal(hull['vertices'],lens['offsets'])
            self.assertGreater(len(hull['triangles']),2)
            self.assertTrue(all(0<=i<len(lens['offsets']) for f in hull['triangles'] for i in f))

    def test_duplicates_do_not_gain_invented_directions_or_facets(self):
        x=np.tile(np.array([1.,0.,0.,0.]),(30,1))
        ids=np.tile(np.arange(1,21),(30,1))
        stats=neighborhood_spectra(x,ids)
        self.assertFalse(stats['valid'].any())
        self.assertEqual(stats['effective'].sum(),0)
        lens=local_lens(x,0,ids[0]);hull=neighborhood_hull(lens)
        self.assertEqual(hull['rank'],0)
        self.assertEqual(hull['triangles'],[])

    def test_mesh_fields_cover_all_points_and_remain_finite(self):
        x=normalize(np.random.default_rng(7).normal(size=(180,10)))
        model=build_map(x,scales=[6,12,24],regions=12,repeats=2)
        atlas=build_atlas(model);atlas['terrain']=enrich_terrain(model,atlas)
        mesh=enrich_mesh(model,atlas)
        self.assertEqual(len(mesh['hidden_two']),len(x))
        self.assertTrue(np.all((mesh['hidden_two']>=0)&(mesh['hidden_two']<=1)))
        self.assertTrue(np.isfinite(mesh['covariance_dimensions']).all())
        for island in atlas['islands']:
            values=island['field']['values']
            self.assertTrue(np.isfinite(values['covariance_height']).all())
            self.assertTrue(np.isfinite(values['hidden_two']).all())
            self.assertEqual(np.shape(values['hidden_two']),np.shape(values['height']))


if __name__=='__main__':unittest.main()
