# Dutch Text Analytics
Dutch Text Analytics is a versatile toolkit designed to facilitate the exploration, execution, and validation of a diverse range of Natural Language Processing (NLP) tasks specifically tailored for the Dutch language. This repository provides a comprehensive set of tools, including code examples, scripts, and resources, to enhance and streamline your Dutch NLP projects.

## Getting Started

### Prerequisites

- Ensure you have Python version 3.8.10 installed on your system.

### Installation

To install the *Dutch Text Analytics* package, open a command prompt and run:

```bash
pip install dutch_text_analytics
```
### Usage

In your Python script or Jupyter Notebook, import the library as follows:

```python
import dutch_text_analytics.text_analytics as ta
```
To access modules and leverage the functionalities provided by the toolkit, instantiate the TextProcessing class:

```python
text_processor = ta.TextProcessing()
```
For detailed usage examples and demonstrations, refer to the demo scripts available in the dutch_text_analytics/demos folder.

## Modules Overview

## Text Processing

The TextProcessing module provides powerful tools for working with Dutch text, including lemmatization, handling separable verbs, and displaying dependency trees.

```python
# Example: Instantiate TextProcessing
text_processor = ta.TextProcessing()

# Example: Lemmatize a sentence
lemmatized_sentence = text_processor.lemmatize("Your Dutch sentence here.")

# Example: Handle separable verbs in a sentence
processed_sentence = text_processor.handle_separable_verbs("Your Dutch sentence here.")

# Example: Display dependency tree of a sentence
text_processor.display_dependency("Your Dutch sentence here.")
```
Explore the module for additional functionalities to enhance your Dutch text processing workflows.

## Demos

Check out the 'dutch_text_analytics/demos' folder for hands-on demonstrations and code examples showcasing the capabilities of Dutch Text Analytics across various NLP tasks.

Feel free to contribute, report issues, or suggest improvements. Happy coding with Dutch Text Analytics!