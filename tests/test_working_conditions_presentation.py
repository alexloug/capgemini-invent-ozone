"""Wilson error bars remain valid for cells with zero or all departures."""
import unittest
import pandas as pd
from src.working_conditions_presentation import rates


class WorkingConditionsPresentationTests(unittest.TestCase):
    def test_rate_intervals_at_zero_and_one(self):
        frame=pd.DataFrame({'group':['A','A','B','B','C','C'],'depart':[0,0,1,1,0,1]})
        result=rates(frame,['group']).set_index('group')
        self.assertEqual(result.loc['A','rate'],0)
        self.assertEqual(result.loc['B','rate'],1)
        self.assertEqual(result.loc['C','rate'],.5)
        self.assertTrue((result.low>=0).all())
        self.assertTrue((result.high<=1).all())
        self.assertTrue((result.rate-result.low>=0).all())
        self.assertTrue((result.high-result.rate>=0).all())


if __name__=='__main__':unittest.main()
