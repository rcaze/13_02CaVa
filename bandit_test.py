#This is the test suite for the bandit script
import numpy as np
import copy
import unittest #Import the testing module
from bandit import * #To import the functions from the bandit

class FirstTestCase(unittest.TestCase):
    def test_gobs(self):
        """Test the generation of observation/reward. Note that only the binomial case is tested"""

        #GIVEN these parameters
        test_cases=[([0.3],1000),([0.1],100)]
        for par, nep in test_cases:
            #WHEN we a set of observation is created
            obs = gobs(par,nep)
            oU = np.sum(np.array(obs)>0)/float(nep)
            if len(par)==1: #testing the case with binomial distribution
                eX = par[0] #We expect that the proportion of ones is
                #THEN the difference between the oU and eX is small for big nep
                self.assertAlmostEqual(oU,eX,delta=0.05)
    
    def oqnext(qprev=[[0.1,0.4],[0.5,0.2]], alphas=[[0.1,0.5],[0.2,0.3]], choice=[0,1], ob=[1,-1]):
        '''The slow non-vectorized qnext'''
        qnext = copy.deepcopy(qprev) #The expected qnext
        for i, qcur in enumerate(qprev):
            if ob[i] >= qcur[choice[i]]:
                qnext[i][choice[i]] = qcur[choice[i]] + alphas[i][0] * ( ob[i] - qcur[choice[i]] )
            else:
                qnext[i][choice[i]] = qcur[choice[i]] + alphas[i][1] * ( ob[i] - qcur[choice[i]] )
        return qnext

   
    def test_qnext(self):
        """Test the generation of the next Q-value"""
        #GIVEN
        test_cases=[([[0.1,0.4],[0.5,0.2]], [[0.1,0.5],[0.2,0.3]], [0,1], [1,-1])]
        for qprev, alphas, choice, ob in test_cases:
            #WHEN
            qnext = vqnext(np.array(qprev), np.array(alphas), np.array(choice), np.array(ob))
            #THEN
            eqnext = copy.deepcopy(qprev) #The expected qnext
            for i, qcur in enumerate(qprev):
                if ob[i] >= qcur[choice[i]]:
                    eqnext[i][choice[i]] = qcur[choice[i]] + alphas[i][0] * ( ob[i] - qcur[choice[i]] )
                else:
                    eqnext[i][choice[i]] = qcur[choice[i]] + alphas[i][1] * ( ob[i] - qcur[choice[i]] )
            np.testing.assert_array_equal(qnext,eqnext)

    def test_equality(self):
        """Docstrings are printed during executions of the tests in the Eclipse IDE"""
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()
