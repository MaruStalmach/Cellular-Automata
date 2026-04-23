from libs.Cells import Cell

# helper func for testing
def generate_number():
    return 42

def test_cell_initialization_zero_values():
    cell = Cell(keys=['alive', 'count'], random=False)

    assert cell['alive'] == 0
    assert cell['count'] == 0

def test_cell_initialization_with_custom_function():
    cell = Cell(keys=['alive', 'count'], random=True, random_func=generate_number)

    assert cell['alive'] == 42
    assert cell['count'] == 42



