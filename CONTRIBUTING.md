# Contributing to the Medical Event Data Standard (MEDS)

Thank you for your interest in contributing to MEDS! We welcome contributions from everyone, whether it's through reporting issues, suggesting improvements, or submitting code.

## How to Contribute

### Reporting Issues

If you find any bugs, inconsistencies, or have feature requests, please:

- Check both open and closed issues in the issues section to ensure your issue hasn't already been addressed or previously discussed.
- Clearly describe the problem, including steps to reproduce and expected vs. actual behavior.
- If reopening a previously closed issue, clearly state why you believe this issue now merits reconsideration.

### Suggesting Enhancements

Enhancements can include improvements to documentation, schema definitions, or overall design:

- Submit your suggestion as a GitHub issue.
- Clearly explain your suggestion, its benefits, and any potential drawbacks or considerations.

### Pull Requests

Pull requests (PRs) are warmly welcomed! New contributions should generally build on and submit PRs to the `dev` branch instead of the `main` branch. We use a release candidate process to manage stable releases from accumulated contributions on the `dev` branch.

#### Guidelines for PRs:

1. **Fork and Clone the Repository**: Start by forking the relevant repository and cloning your fork locally.

    ```sh
    git clone https://github.com/your-username/MEDS_extract.git
    ```

2. **Create a Branch**: For each contribution, create a dedicated branch with a descriptive name based on the `dev` branch:

    ```sh
    git checkout dev
    git checkout -b my-new-feature
    ```

3. **Make Changes**: Implement your changes clearly and concisely.

    - Follow existing code and documentation style.
    - Include clear commit messages describing each step of your changes.

4. **Testing and Code Style**:

    - MEDS uses automated workflows for testing and pre-commit code style checks. PRs must pass these checks to be accepted.
    - You can install the necessary development dependencies locally with:
        ```sh
        pip install -e .[dev]
        ```
    - Optionally, you can also install test dependencies:
        ```sh
        pip install -e .[tests]
        ```
    - After installation, set up pre-commit hooks manually:
        ```sh
        pre-commit install
        ```
    - Ensure your code passes all checks locally before submitting your PR to shorten development iterations.

5. **Update Documentation**: Always update documentation alongside your changes. If you modify schemas or dataset structures, reflect those changes clearly in the relevant README sections and schema documentation.

    - Docstrings should follow the Google style guide and include doctests where appropriate.

6. **Submit Your PR**: Push your changes to your fork and submit a PR:

    ```sh
    git push origin my-new-feature
    ```

### Code and Documentation Style

- Follow PEP 8 style guidelines for Python code.
- Use Google-style docstrings with doctests where applicable.
- Clearly document new functions, methods, or schema changes.
- Ensure consistency with existing schema definitions.
- Make sure you clearly document the current base version of MEDS data that the tool is compatible with.

### Review Process

- All PRs will undergo review by repository maintainers.
- Be prepared to discuss feedback and make revisions if requested.
- Maintainers aim to respond to PRs within one week.

## Community and Communication

We encourage active discussions and community involvement! Feel free to participate in discussions on issues or reach out to maintainers for questions.

## Code of Conduct

All contributors are expected to adhere to a respectful and inclusive Code of Conduct. Please ensure your interactions remain professional, constructive, and welcoming.

Thank you for contributing to MEDS! We value your input and support.
