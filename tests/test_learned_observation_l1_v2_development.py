import importlib.util
from pathlib import Path
import unittest
import numpy as np
P=Path(__file__).parents[1]/'experiments'/'run_learned_observation_l1_v2_development.py'; S=importlib.util.spec_from_file_location('v2dev',P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)
class TestV2Development(unittest.TestCase):
 def test_a1_median_mad_and_clip(self):
  cfg={'normalization':{'mad_scale':1.4826,'mad_floor':1e-6,'clip_min':-6.,'clip_max':6.}}
  x=np.tile(np.arange(13,dtype=float)[:,None],(1,30)); z=M.transform(x,'A1',cfg); self.assertTrue(np.allclose(np.median(z,axis=0),0)); self.assertLessEqual(abs(z).max(),6)
 def test_a2_boundary_and_length(self):
  cfg={'normalization':{'mad_scale':1.,'mad_floor':1e-6,'clip_min':-100.,'clip_max':100.}}
  x=np.tile(np.arange(13,dtype=float)[:,None],(1,30)); y=M.transform(x,'A2',cfg); self.assertEqual(y.shape,(13,30)); self.assertTrue(np.allclose(y[1:-1],0)); self.assertTrue(np.allclose(y[0],-y[-1]))
if __name__=='__main__': unittest.main()
