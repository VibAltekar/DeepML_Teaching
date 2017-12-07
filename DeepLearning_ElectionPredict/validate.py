import json
import os

possible_answers = set(['Barack Obama', 'Mitt Romney'])

def run_basic_validations(validation_file):

    assert os.path.exists(validation_file), 'Prediction file predictions.csv is missing'

    with open(validation_file, 'rt') as pred_file:
        lines = pred_file.readlines()
        lines = [l.strip() for l in lines]

        assert lines[0] == 'Winner', 'Missing header row Winner: received %s' % lines[0]

        assert len(lines) == 1902, 'Did not submit correct number of predictions'

        sample = lines[123]

        assert sample in possible_answers

    print('Formatting looks good, hope your model performs well!')


if __name__ == '__main__':
    run_basic_validations('predictions.csv')
