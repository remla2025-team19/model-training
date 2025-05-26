from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker
import astroid

class RandomnessUncontrolledChecker(BaseChecker):
    __implements__ = IAstroidChecker
    name = 'randomness-uncontrolled'
    priority = -1
    # Custom message
    msgs = {
        'W9003': (
            'Randomness used without seed.', # User-facing
            'randomness-uncontrolled',
            'Use random.seed() or np.random.seed() to ensure reproducibility.', # Description
        ),
    }

    def __init__(self, linter):
        super().__init__(linter)
        # By default seed has not been set
        self.seed_set = False

    def visit_call(self, node):
        # Check if func is an attribute
        if not isinstance(node.func, astroid.Attribute):
            return
        
        # If attribute detected as seed, set flag True
        attr = node.func.attrname
        if attr == 'seed':
            self.seed_set = True
            return

        # Random function called wo seed, issue warning
        if attr in ('shuffle', 'choice', 'rand', 'randint') and not self.seed_set:
            self.add_message('randomness-uncontrolled', node=node)

    def leave_module(self, _):
        self.seed_set = False

class DataLeakageChecker(BaseChecker):
    __implements__ = IAstroidChecker
    name = 'data-leakage-checker'
    priority = -1
    msgs = {
        'W9004': (
            'Data leakage possibility since fitting done before splitting',
            'data-leakage-detected',
            'Ensure train/test split is done before fitting.',
        ),
    }

    def __init__(self, linter):
        super().__init__(linter)
        self.split_seen = False

    def visit_call(self, node):
        # Check if function is an attribute
        if not isinstance(node.func, astroid.Attribute):
            return

        attr = node.func.attrname

        # If split encountered, update flag to True
        if attr == 'train_test_split':
            self.split_seen = True
            return

        # By the time data fit, split already changed to True ideally, if not:
        if attr == 'fit_transform' and not self.split_seen:
            self.add_message('data-leakage-detected', node=node)


    def leave_module(self, _):
        self.split_seen = False

# Register both
def register(linter):
    linter.register_checker(RandomnessUncontrolledChecker(linter))
    linter.register_checker(DataLeakageChecker(linter))
