from gpu_models import lightgbm_gpu,xgboost_gpu,catboost_gpu,nn_gpu
import unittest
import pandas as pd


class TestLGBCPU(unittest.TestCase):
    def setUp(self):
        self.cls = lightgbm_gpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
class TestXGBCPU(unittest.TestCase):
    def setUp(self):
        self.cls = xgboost_gpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
class TestCATBCPU(unittest.TestCase):
    def setUp(self):
        self.cls = catboost_gpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
class TestNNCPU(unittest.TestCase):
    def setUp(self):
        self.cls = nn_gpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
        
        
if __name__=='__main__':
    unittest.main()