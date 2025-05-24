"""Custom Pylint checkers for ML code smells."""

from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker
import astroid


class UnnecessaryIterationChecker(BaseChecker):
    """
    Checks for unnecessary iteration in Pandas/NumPy contexts,
    suggesting vectorized alternatives.
    """

    __implements__ = IAstroidChecker
    name = 'unnecessary-iteration-checker'
    priority = -1
    msgs = {
        'W9001': (
            'Unnecessary iteration: Found loop of type "for i in range(len(variable))". '
            'Consider using vectorized operations if "variable" is a Pandas DataFrame/Series.',
            'unnecessary-iteration',
            'Advise using vectorized operations instead of explicit loops over range(len(data_structure)).',
        ),
    }

    def visit_for(self, node: astroid.For) -> None:
        """
        Called when a 'for' loop is visited.
        Detects 'for i in range(len(variable))' pattern.
        """
        if not isinstance(node.iter, astroid.Call):
            return
        if (
            not isinstance(node.iter.func, astroid.Name)
            or node.iter.func.name != 'range'
        ):
            return
        if len(node.iter.args) != 1:
            return
        range_arg = node.iter.args[0]
        if not isinstance(range_arg, astroid.Call):
            return
        if not isinstance(range_arg.func, astroid.Name) or range_arg.func.name != 'len':
            return
        if len(range_arg.args) == 1:
            self.add_message('unnecessary-iteration', node=node)


class PyTorchCallableMisuseChecker(BaseChecker):
    """
    Checks for direct calls to 'self.module.forward()' or 'module.forward()'.
    Recommends using 'self.module()' or 'module()' instead, as this ensures PyTorch hooks are executed.
    """

    __implements__ = IAstroidChecker
    name = 'pytorch-forward-misuse-checker'
    priority = -1
    msgs = {
        'W9002': (
            "Direct call to 'self.module.forward()' or 'module.forward()' detected. "
            "It is recommended to use 'self.module()' or 'module()' instead, as this ensures all PyTorch hooks are executed.",
            'pytorch-forward-misuse',
            "Advises using 'self.module()' or 'module()' over 'self.module.forward()' or 'module.forward()' for PyTorch nn.Modules.",
        ),
    }

    def visit_call(self, node: astroid.Call) -> None:
        """
        Called when a function call is visited.
        Detects 'self.module.forward()' or 'module.forward()' patterns.
        """
        # Ensure the called function is an attribute access, e.g., obj.method
        if not isinstance(node.func, astroid.Attribute):
            return

        # Ensure the attribute being called is 'forward'
        if node.func.attrname != 'forward':
            return

        # 'base_expr' is the expression on which '.forward' is called.
        # e.g., for 'self.model.forward()', base_expr is 'self.model' (astroid.Attribute)
        # e.g., for 'model.forward()', base_expr is 'model' (astroid.Name)
        base_expr = node.func.expr

        # Check for 'self.some_attribute.forward()'
        if isinstance(base_expr, astroid.Attribute):
            # base_expr.expr is the part before the dot, e.g., 'self'
            if isinstance(base_expr.expr, astroid.Name) and base_expr.expr.name == 'self':
                self.add_message('pytorch-forward-misuse', node=node)
        # Check for 'some_name.forward()'
        elif isinstance(base_expr, astroid.Name):
            self.add_message('pytorch-forward-misuse', node=node)


def register(linter):
    """This required method auto registers the checkers."""
    linter.register_checker(UnnecessaryIterationChecker(linter))
    linter.register_checker(PyTorchCallableMisuseChecker(linter))
