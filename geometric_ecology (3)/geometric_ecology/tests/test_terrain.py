"""Check terrain metaphors against actual measured geometry and controls."""
import unittest
import numpy as np
from geometric_ecology.core import build_map,sparse_graph,normalize
from geometric_ecology.atlas import build_atlas
from geometric_ecology.terrain import route_set,enrich_terrain
from geometric_ecology.terrain_lab import build_lab_data

class TerrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.lab=build_lab_data()

    def test_dimension_changes_with_matched_radius(self):
        low,high=self.lab['vectors']['2'],self.lab['vectors']['12']
        self.assertAlmostEqual(low['radius'],high['radius'],places=6)
        self.assertGreater(low['visible_two_fraction'],.99)
        self.assertLess(high['visible_two_fraction'],.4)
        self.assertGreater(high['participation_ratio'],3*low['participation_ratio'])
        self.assertEqual(low['components90'],2)
        self.assertGreaterEqual(high['components90'],8)
        self.assertEqual(len(high['direction_scores'][0]),12)

    def test_route_entropy_is_not_neighbor_degree(self):
        star=sparse_graph(5,np.array([[0,1],[0,2],[0,3],[0,4]]),np.ones(4))
        self.assertEqual(route_set(star,1,2)['effective_routes'],1)
        diamond=sparse_graph(4,np.array([[0,1],[1,3],[0,2],[2,3]]),np.ones(4))
        result=route_set(diamond,0,3)
        self.assertAlmostEqual(result['entropy_bits'],1)
        self.assertAlmostEqual(result['effective_routes'],2)
        self.assertEqual(len(result['paths']),2)

    def test_maze_and_tree_have_different_actual_routes(self):
        maze=self.lab['graphs']['maze'];tree=self.lab['graphs']['tree']
        np.testing.assert_array_equal(maze['xy'],tree['xy'])
        self.assertEqual(maze['routes']['costs'][0],tree['routes']['costs'][0])
        self.assertTrue(maze['routes']['capped'])
        self.assertAlmostEqual(maze['routes']['effective_routes'],12)
        self.assertAlmostEqual(tree['routes']['effective_routes'],1)
        for scene in [maze,tree]:
            edges={tuple(sorted(e)) for e in scene['edges']}
            for path,cost in zip(scene['routes']['paths'],scene['routes']['costs']):
                self.assertEqual(len(path),len(set(path)))
                self.assertTrue(all(tuple(sorted((a,b))) in edges for a,b in zip(path,path[1:])))
                self.assertAlmostEqual(cost,len(path)-1)

    def test_gap_bottleneck_and_isolation_are_distinct(self):
        self.assertGreater(self.lab['graphs']['cliff']['gap_ratio'],4)
        self.assertLess(self.lab['graphs']['pass']['gap_ratio'],1.5)
        self.assertFalse(self.lab['graphs']['island']['routes']['reachable'])
        self.assertTrue(all(s['conductance']<.1 for s in self.lab['graphs']['ridge']['scales']))

    def test_smooth_ring_is_not_given_artificial_ridges(self):
        angle=np.linspace(0,2*np.pi,300,endpoint=False)
        x=normalize(np.column_stack([np.ones(300),.2*np.cos(angle),.2*np.sin(angle),np.zeros(300)]))
        model=build_map(x,scales=[6,12,24],regions=12,repeats=2)
        atlas=build_atlas(model);terrain=enrich_terrain(model,atlas)
        self.assertEqual(terrain['ridges'],[])
        self.assertTrue(np.isfinite(terrain['height']).all())
        self.assertTrue(np.all((terrain['fog']>=0)&(terrain['fog']<=1)))
        for lens in terrain['lenses'].values():self.assertAlmostEqual(sum(lens['spectrum']),1,places=5)

if __name__=='__main__':unittest.main()
