"""The atlas must expose computed structure without inventing graph connections."""
import base64
import unittest
import numpy as np
from geometric_ecology.core import build_map, sparse_graph
from scipy.sparse.csgraph import connected_components
from geometric_ecology.demo import synthetic_corpus
from geometric_ecology.atlas import build_atlas


class AtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x,ids,_=synthetic_corpus(600,24,42)
        cls.model=build_map(x,ids,scales=[6,12,24],regions=24,repeats=2)
        cls.atlas=build_atlas(cls.model)

    def test_all_points_are_represented_with_finite_positions(self):
        a,m=self.atlas,self.model
        self.assertEqual(a['xy'].shape,(len(m['x']),2))
        self.assertTrue(np.isfinite(a['xy']).all())
        total=sum(i['n'] for i in a['islands'])+a['small_component_n']
        self.assertEqual(total,len(m['x']))
        self.assertTrue(np.all((a['fidelity']>=0)&(a['fidelity']<=1)))

    def test_disconnected_island_discs_do_not_overlap(self):
        islands=self.atlas['islands']
        for i,a in enumerate(islands):
            inds=self.model['component']==a['component']
            distances=np.linalg.norm(self.atlas['xy'][inds]-a['center'],axis=1)
            self.assertLessEqual(distances.max(),a['radius']+1e-7)
            for b in islands[i+1:]:
                distance=np.linalg.norm(np.array(a['center'])-b['center'])
                self.assertGreater(distance,a['radius']+b['radius'])
        if self.atlas['small_component_box']:
            box=self.atlas['small_component_box']
            for component in self.atlas['minor_components']:
                positions=self.atlas['xy'][self.model['component']==component]
                self.assertTrue(np.all(positions[:,0]>=box[0]))
                self.assertTrue(np.all(positions[:,0]<=box[2]))

    def test_every_road_uses_actual_edges_and_its_own_crossing(self):
        m=self.model
        for road in self.atlas['roads']:
            path=road['path']
            self.assertEqual(m['region'][path[0]],road['a'])
            self.assertEqual(m['region'][path[-1]],road['b'])
            cost=0
            crossing=0
            for u,v in zip(path,path[1:]):
                self.assertGreater(float(m['graph'][u,v]),0)
                self.assertEqual(m['component'][u],m['component'][v])
                cost+=float(m['graph'][u,v])
                if m['region'][u]!=m['region'][v]:crossing+=1
            self.assertEqual(crossing,1)
            self.assertAlmostEqual(cost,road['cost'],places=6)
            if road['narrow']:
                self.assertTrue(road['regional_cut'])
                self.assertLess(road['conductance'],.04)
                self.assertGreater(road['centrality_percentile'],.6)

    def test_marked_regional_cuts_disconnect_the_original_graph(self):
        m=self.model
        checked=0
        edges=m['stable_edges']
        endpoints=m['region'][edges]
        for road in self.atlas['roads']:
            if not road['regional_cut']:continue
            a,b=road['a'],road['b']
            keep=~(((endpoints[:,0]==a)&(endpoints[:,1]==b))|((endpoints[:,0]==b)&(endpoints[:,1]==a)))
            graph=sparse_graph(len(m['x']),edges[keep],np.ones(int(keep.sum())))
            _, labels=connected_components(graph,directed=False)
            self.assertNotEqual(labels[road['path'][0]],labels[road['path'][-1]])
            checked+=1
        self.assertGreater(checked,0)

    def test_rendered_surface_is_embedded_numeric_raster(self):
        for island in self.atlas['islands']:
            image=island['surface']['image']
            self.assertTrue(image.startswith('data:image/png;base64,'))
            self.assertTrue(base64.b64decode(image.split(',')[1]).startswith(b'\x89PNG\r\n\x1a\n'))
            self.assertTrue(island['surface']['coasts'])

if __name__=='__main__':unittest.main()
