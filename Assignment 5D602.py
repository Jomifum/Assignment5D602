#Assignment 5D602
#By Jose Fuentes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import unittest
import requests
import json


def exercise01():
    '''
    Create a DataFrame df with 4 columns and 3 rows of data in one line of code. The data can be arbitrary integers:
    '''
    # ------ Place code below here \/ \/ \/ ------
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [7, 8, 9, 10]])
    # ------ Place code above here /\ /\ /\ ------
    return df


def exercise02(a):
    # Convert the list to a numpy ndarray called array.
    # ------ Place code below here \/ \/ \/ ------
    array = np.array(a)
    # ------ Place code above here /\ /\ /\ ------
    return array


def exercise03(a):
    # Return the sum of integers in the ndarray using NumPy.
    # ------ Place code below here \/ \/ \/ ------
    sum = np.sum(a)
    # ------ Place code above here /\ /\ /\ ------
    return sum


def exercise04(a):
    # Return the sum of the 2nd column in the ndarray.
    # ------ Place code below here \/ \/ \/ ------
    sum = np.sum(a[:, 1])
    # ------ Place code above here /\ /\ /\ ------
    return sum


def exercise05(n):
    # Return an ndarray filled with zeros of size n x n.
    # ------ Place code below here \/ \/ \/ ------
    zeros = np.zeros((n, n))
    # ------ Place code above here /\ /\ /\ ------
    return zeros


def exercise06(n):
    # Return an ndarray filled with ones of size n x n.
    # ------ Place code below here \/ \/ \/ ------
    ones = np.ones((n, n))
    # ------ Place code above here /\ /\ /\ ------
    return ones


def exercise07(sd, m, s):
    # Return an ndarray of s random numbers with the specified standard deviation and mean.
    # ------ Place code below here \/ \/ \/ ------
    random_numbers = np.random.normal(m, sd, s)
    # ------ Place code above here /\ /\ /\ ------
    return random_numbers


def exercise08():
    '''
    Load the CSV data from URL, return row count, avg_sq_ft, DataFrame with zip 95670, DataFrame without zip 95610.
    '''
    # ------ Place code below here \/ \/ \/ ------
    df = pd.read_csv("https://tinyurl.com/y63q7okz")
    row_count = len(df)
    avg_sq_ft = df['sq__ft'].mean()
    df_zip_95670 = df[df['zip'] == 95670]
    df_zip_not_95610 = df[df['zip'] != 95610]
    # ------ Place code above here /\ /\ /\ ------
    return df, row_count, avg_sq_ft, df_zip_95670, df_zip_not_95610


def exercise10(n):
    # Create a numpy identity matrix of size n.
    # ------ Place code below here \/ \/ \/ ------
    identity_matrix = np.identity(n)
    # ------ Place code above here /\ /\ /\ ------
    return identity_matrix


def exercise11(n):
    '''
    Create a single dimension array, reshape to n/3 columns and 3 rows.
    '''
    # ------ Place code below here \/ \/ \/ ------
    array_1d = np.arange(n)
    array_reshaped = array_1d.reshape((3, n // 3))
    # ------ Place code above here /\ /\ /\ ------
    return array_1d, array_reshaped


def exercise12(n):
    '''
    Create a checkerboard matrix of size 2n x 2n.
    '''
    # ------ Place code below here \/ \/ \/ ------
    checkerboard_matrix = np.tile([[1, 0], [0, 1]], (n, n))
    # ------ Place code above here /\ /\ /\ ------
    return checkerboard_matrix


def exercise13(n):
    '''
    Create a pandas Series with random integers, cumulative sum, plot with date index.
    '''
    # ------ Place code below here \/ \/ \/ ------
    date_index = pd.date_range(start="1/1/2010", periods=n)
    s = pd.Series(np.random.randint(0, n, size=n), index=date_index).cumsum()
    s.plot(title='Cumulative Sum over Time')
    plt.show()
    # ------ Place code above here /\ /\ /\ ------
    return s


def exercise14(words):
    '''
    Create and return a DataFrame that tabulates the length of each word in the list.
    '''
    # ------ Place code below here \/ \/ \/ ------
    df = pd.DataFrame({'word_length': list(map(len, words))})
    # ------ Place code above here /\ /\ /\ ------
    return df


def exercise15():
    '''
    Extract every 5th row with street address and zip code columns from the DataFrame in exercise08.
    '''
    # ------ Place code below here \/ \/ \/ ------
    df, _, _, _, _ = exercise08()
    df = df.iloc[::5][['street', 'zip']]
    # ------ Place code above here /\ /\ /\ ------
    return df


class TestAssignment5(unittest.TestCase):
    def test_exercise15(self):
        print('Testing exercise 15')
        df = exercise15()
        print(df)
    
    def test_exercise14(self):
        print('Testing exercise 14')
        df = exercise14(['cat', 'frog', 'walrus', 'antelope'])
        print(df)

    def test_exercise13(self):
        print('Testing exercise 13')
        s = exercise13(1000)
        self.assertEqual(s.index[0], pd.Timestamp('2010-01-01 00:00:00'))
        self.assertEqual(len(s.index), 1000)

    def test_exercise12(self):
        print('Testing exercise 12')
        cm = exercise12(10)
        self.assertEqual(cm.shape[0], 20)
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[0, 1], 0)
        cm = exercise12(5)
        self.assertEqual(cm.shape[0], 10)
        self.assertEqual(cm[0, 0], 1)
        self.assertEqual(cm[0, 1], 0)

    def test_exercise11(self):
        print('Testing exercise 11')
        a1d, ar = exercise11(15)
        self.assertEqual(a1d.shape[0], 15)
        self.assertEqual(ar.shape[0], 3)
        self.assertEqual(ar.shape[1], 5)

    def test_exercise10(self):
        print('Testing exercise 10')
        im = exercise10(10)
        self.assertEqual(im.shape[0], 10)
        self.assertEqual(im.shape[1], 10)

    def test_exercise08(self):
        print('Testing exercise 8')
        df, row_count, avg_sq_ft, df_zip_95670, df_zip_not_95610 = exercise08()
        self.assertEqual(df.shape[0], 985)
        self.assertEqual(df.shape[1], 12)
        self.assertEqual(row_count, 985)
        self.assertAlmostEqual(avg_sq_ft, 1314.91675127, 2)
        self.assertEqual(df_zip_95670.shape[0], 21)
        self.assertEqual(df_zip_not_95610.shape[0], 978)

    def test_exercise07(self):
        print('Testing exercise 7')
        z = exercise07(10, 5, 100000)
        self.assertEqual(z.shape[0], 100000)
        self.assertLessEqual(np.average(z), 5.2)
        self.assertGreaterEqual(np.average(z), 4.7)
        z = exercise07(5, 10, 100000)
        self.assertEqual(z.shape[0], 100000)
        self.assertLessEqual(np.average(z), 10.2)
        self.assertGreaterEqual(np.average(z), 9.7)

    def test_exercise06(self):
        print('Testing exercise 6')
        z = exercise06(7).shape
        self.assertEqual(z[0], 7)
        self.assertEqual(z[1], 7)
        z = exercise05(70).shape
        self.assertEqual(z[0], 70)
        self.assertEqual(z[1], 70)

    def test_exercise05(self):
        print('Testing exercise 5')
        z = exercise05(7).shape
        self.assertEqual(z[0], 7)
        self.assertEqual(z[1], 7)
        z = exercise05(70).shape
        self.assertEqual(z[0], 70)
        self.assertEqual(z[1], 70)

    def test_exercise04(self):
        print('Testing exercise 4')
        array = np.array([[1, 1, 1, 1, 1], [0, 2, 0, 0, 1]])
        sum = exercise04(array)
        self.assertEqual(sum, 3)
        array = np.array([[1, 6, 1, 1, 1], [0, 2, 0, 0, 1]])
        sum = exercise04(array)
        self.assertEqual(sum, 8)

    def test_exercise03(self):
        print('Testing exercise 3')
        self.assertEqual(exercise03(np.array([1, 2, 3])), 6)
        self.assertEqual(exercise03(np.array([1, 2, 3, 4, 5])), 15)

    def test_exercise02(self):
        print('Testing exercise 2')
        self.assertEqual(exercise02([1, 2, 3]).tolist(), [1, 2, 3])
        self.assertEqual(exercise02([1, 2, 3, 4, 5]).tolist(), [1, 2, 3, 4, 5])

    def test_exercise01(self):
        print('Testing exercise 1')
        df = exercise01()
        self.assertEqual(df.shape[0], 3)
        self.assertEqual(df.shape[1], 4)


if __name__ == '__main__':
    unittest.main()
