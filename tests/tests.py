import torch
#
# def test_fi():
#     input = torch.LongTensor([[0, 0], [0, 1], [1, 0], [1, 1]])
#     expected = torch.LongTensor([0, 1, 2, 3])
#
#     actual = gaussian.fi(input, (2,2))
#
#     print(actual)
#
#
# def test_fi_mat():
#     input = torch.LongTensor([[[0, 0], [0, 1], [1, 0], [1, 1]]])
#     expected = torch.LongTensor([0, 1, 2, 3])
#
#     actual = gaussian.fi_matrix(input, torch.LongTensor((2, 2)))
#
#     print(actual)
#
# def test_sort():
#     indices = torch.LongTensor([[[6, 3], [1, 2]], [[5, 8], [1, 3]]])
#     vals = torch.FloatTensor([[0.1, 0.2], [0.3, 0.4]])
#
#     hyper.sort(indices, vals)
#
#     print(indices)
#     print(vals)
#
#
#
# if __name__ == '__main__':
#     # unittest.main()
#
#     test_fi()
#     test_fi_mat()