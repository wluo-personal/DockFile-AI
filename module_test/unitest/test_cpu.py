from cpu_models import lightgbm_cpu,xgboost_cpu,catboost_cpu,nn_cpu
import unittest
import pandas as pd


class TestLGBCPU(unittest.TestCase):
    def setUp(self):
        self.cls = lightgbm_cpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
class TestXGBCPU(unittest.TestCase):
    def setUp(self):
        self.cls = xgboost_cpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
class TestCATBCPU(unittest.TestCase):
    def setUp(self):
        self.cls = catboost_cpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
class TestNNCPU(unittest.TestCase):
    def setUp(self):
        self.cls = nn_cpu()
        self.df = pd.read_csv('../../data/pima-indians-diabetes.data.csv')
        self.x = self.df.iloc[:,:8].values
        self.y = self.df.iloc[:,8].values
        
    def test_fit(self):
        self.assertTrue(self.cls.fit(self.x,self.y))
        
        
        
if __name__=='__main__':
    unittest.main()