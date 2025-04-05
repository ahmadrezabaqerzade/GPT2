<div align="center">
    <img src="https://github.com/user-attachments/assets/c9771160-380c-441f-baf2-cb8845eb6072" alt="Logo" width="" height="200">
  </a>

<h1 align="center"></h1>
</div>

# GPT-2: Overview
GPT-2 (Generative Pre-trained Transformer 2) is a large language model developed by OpenAI in 2019. It is the successor to GPT-1 and is based on the Transformer architecture (specifically the decoder-only variant). GPT-2 was notable for its ability to generate coherent and contextually relevant text, making it a breakthrough in natural language processing (NLP).

**Key Features of GPT-2:**
   * **Model Size Variants: GPT-2 was released in multiple sizes:**

        * **Small** (117M parameters)

        * **Medium** (345M parameters)

        * **Large** (774M parameters)

        * **Extra Large (XL)** (1.5B parameters)

  * **Unsupervised Learning**: Pre-trained on a massive corpus of internet text (WebText) in a self-supervised manner (predicting the next word in a sequence).

  * **Fine-Tuning Capability**: Could be fine-tuned for specific tasks (e.g., translation, summarization).

  * **Controversy**: Initially, OpenAI withheld the full 1.5B model due to concerns about misuse (e.g., fake news generation), but later released it.

**Architecture**:
  * Based on **Transformer decoder blocks** (no encoder).

  * Uses **masked self-attention** (causal attention) to prevent looking ahead in the sequence.

  * Trained using **next-word prediction** (autoregressive modeling).
