# Logits and Perplexity Extension Design Document

## Overview

This document outlines the plan for extending the existing LLM text-webui with a new feature: a table displaying logits and perplexity values during text generation. Building upon existing extensions that color generated text based on token perplexity and probability, and display logits for the next token with and without sampling, this new feature aims to provide a comprehensive view of the model's decision-making process.

## Goals

1. Collect and display logits and perplexity values in real-time during text generation.
2. Leverage existing extensions, reusing their functionality where applicable.
3. Provide a clear and informative visual representation of the model's output.

## To-Do List

### Extraction and Collection

1. **Extract logit collection**: Move the functionality responsible for collecting logit values and updating the UI during generation from the existing coloring extension.
2. **Gather logits before and after sampling**: Utilize the existing interface functions to retrieve logits both before and after applying sampling.

### Storage and Display

3. **Store collected values**: Determine a suitable data structure to hold the collected logits and perplexity values, allowing for efficient appending during generation.
4. **Display information**: Design a user-friendly table or visualization to present the collected data, ensuring easy comprehension of the model's behavior.

### Additional Considerations

5. **Performance optimization**: Ensure the extension does not negatively impact the overall performance of the webui, particularly during long-generation tasks.
6. **User customization**: Allow users to configure the display settings, such as choosing which values to show or adjusting the table layout.
7. **Documentation and testing**: Thoroughly document the extension's implementation and conduct comprehensive testing to guarantee its stability and accuracy.
