# Contributing

Thanks for your interest in contributing to pycompete. We welcome any contributions deemed of use to the CrunchDAO ML competition.

## How to submit a contribution

1. Clone this repository and create a new branch.
2. Make the changes on your local branch.
  a) Add/refactor unit tests as appropiate.
  b) Run `make lint-fix`, commit any changes.
3. Run `make test`, ensure all tests pass. 
4. Submit a pull request.

We'll attempt to review your contribution as soon as possible.

## Linting
We use [`black`](https://github.com/psf/black) for linting.

## Testing 

We use `pytest` to manage our testing code, we prefer a functional approach but don't strictly enforce it.
