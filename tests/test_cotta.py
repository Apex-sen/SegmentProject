import unittest

import torch
from torch import nn

from ctta.cotta import CoTTA


class CoTTATests(unittest.TestCase):
    def test_student_learns_teacher_probabilities_when_classes_agree(self):
        model = nn.Conv2d(1, 2, kernel_size=1)
        with torch.no_grad():
            model.bias.copy_(torch.tensor([2.0, 0.0]))

        adapt = CoTTA(model, learning_rate=0.1, confidence_threshold=0.0, restore_probability=0.0)
        with torch.no_grad():
            adapt.teacher.bias[0] = 1.0

        image = torch.zeros(1, 1, 2, 2)
        valid = torch.ones(1, 2, 2, dtype=torch.bool)
        with torch.no_grad():
            teacher_prediction = adapt.teacher(image).softmax(1)
            before = model(image).softmax(1)
        self.assertTrue(torch.equal(before.argmax(1), teacher_prediction.argmax(1)))

        prediction = adapt(image, valid)

        with torch.no_grad():
            after = model(image).softmax(1)
        self.assertGreater(after[0, 0, 0, 0].item(), teacher_prediction[0, 0, 0, 0].item())
        self.assertLess(after[0, 0, 0, 0].item(), before[0, 0, 0, 0].item())
        torch.testing.assert_close(prediction, teacher_prediction)
