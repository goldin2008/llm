def addition(num1, num2=0):
    """
    Adds two numbers

    Args:

    num1: The first number

    num2: The second number, default 0

    Returns:

    The result of the addition process
    """

    return num1 + num2


def mult(num1, num2=1):
    """
    Multiplies two numbers

    Args:

    `num1`: The first number

    `num2`: The second number, default `1`

    Returns:

    The result of the **multiplication** process
    """

    """
    ...
    Handles exception by a check,

        ```python
        if num2 != 0:
            return (num1/num2)
        else:
            raise ValueError('The second argument cannot be zero')
        ```

     """

    return num1 * num2


def pythagorus(num1, num2=0):
    """
    ### Description:

    Calculates the root-sum-of-square of two numbers

    ### Args:

    `num1`: The first number

    `num2`: The second number, default `0`

    ### Returns:

    The result of the **pythagorus** formula

    $$dist = \\sqrt { a^2+b^2 }$$
    """
