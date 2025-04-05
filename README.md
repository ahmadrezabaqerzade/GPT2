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

# Dataset
## TinyStories Dataset
The TinyStories dataset is a synthetic, simplified text corpus designed for training and evaluating small language models. It was introduced to help researchers experiment with lightweight models that can still generate meaningful text.

**Key Features of TinyStories**:
 * **Simple Language**: Contains short stories with basic vocabulary and grammar, making it easier for small models to learn.

 * **Focused on Coherence**: Designed to test whether small models can maintain narrative consistency (unlike random text generation).

 * **Use Case**: Helps in studying how small transformers (1M–100M parameters) perform compared to large models like GPT-2.

**Why TinyStories?**
 * Large models like GPT-2 require massive computational resources.

 * TinyStories allows testing **fundamental language understanding** without needing huge models.

 * Useful for **educational and research purposes** in low-resource settings.

**Example TinyStories Text**:

<details>
<summary>📖 **The Story of Max the Clever Dog**</summary>
    
>Once upon a time, there was a clever little dog named Max. Max loved to run and play with his friends in the park. One day, Max was running very fast when he fell and hurt his knee.
Max went to his friend, the wise old owl, and said, "Owl, my knee hurts. What can I do?" The owl thought for a moment and said, "Max, you should test your knee. Try to walk slowly and see if it still hurts."
So Max tested his knee by walking slowly. At first, it hurt a little, but soon Max felt better. He said, "Thank you, Owl, for your help. Now I can play with my friends again."
Max was so happy that he could play with his friends without pain. He learned that sometimes, it was good to slow down and listen to his body. And Max and his friends played happily in the park ever after.

</details>

**the simple structure, making it ideal for small models.**

## Dataset Analysis

Number of characters in train part:  **1.902088781 Billion characters**

Number of tokens in train part:  **439.039906 Million tokens**

Number of unique tokens in train part:  **63577 tokens**

On average, there are **897 characters** in **each story**.

On average, there are **207 tokens** in **each story**.

The most common token is **'.'** with **36.459483 million occurrences**.

The **mean** token repetition count is **6905**.

The **standard deviation** of token repetition is **226804**.

The **minimum** token repetition count is **1**.

The **25th** percentile of token repetition is **1**.

The **median** token repetition count is **3**.

The **75th** percentile of token repetition is **24**.

The **maximum** token repetition count is **36459483**.

📊 Text Dataset Statistics

--------------------------------------------------

• Characters: 1.90 Billion

• Tokens: 439.04 Million

• Unique Tokens: 63,577

📝 Per-Story Averages:

  → Characters: 897
  
  → Tokens: 207

🏆 Most Frequent Token:

  → '.' (36.46 million occurrences)

📈 Token Repetition Statistics:

  → Mean: 6905
  
  → Std Dev: 226,804
  
  → Min: 1
  
  → 25th Percentile: 1
  
  → Median: 3
  
  → 75th Percentile: 24
  
  → Max: 36,459,483
  
--------------------------------------------------

**🔝 Top 10 Tokens:**

| token  | count |
| ------------- | ------------- |
| .  | 36,459,483  |
| the  | 20,239,799  |
| and  | 18,112,895  |
| ,  | 17,359,980  |
| to  | 12,648,384  |
| a  | 11,475,893  |
| was  | 9,436,658  |
| he  | 7,977,807  |
| she  | 7,710,551  |
| it  | 7,104,640  |

![top10token](https://github.com/user-attachments/assets/26ae4c4f-42e1-4a97-99f7-4cdf443df572)

**Token Frequency Distribution:**

![token-freq-dist](https://github.com/user-attachments/assets/28c9d932-3562-45e2-8d79-2679ff4dede0)

**Token Frequency Distribution(BoxPlot):**

![boxplot-dist](https://github.com/user-attachments/assets/a8864717-ff62-4b64-9b57-2d20410264bd)

**📊 Statistical Analysis of Token Distribution**

**1. Skewness (100.6):**

* Interpretation: Extremely right-skewed distribution

* Indicates:

  * Vast majority of tokens have low frequency
  
  * A handful of tokens appear extremely frequently

  * Typical in natural language (few common words, many rare words)

**2. Kurtosis (13,057.8):**

* Interpretation: Leptokurtic distribution with heavy tails

* Shows:
  
  * Sharp peak at lower frequencies

  * Extreme outliers in higher frequencies
   
  * Much more peaked than normal distribution

**3. Gini Coefficient (0.99):**

* Interpretation: Extreme inequality in token frequency

* Means:
  
  * Nearly all frequency concentrated in very few tokens
   
  * Similar to wealth distribution in unequal economies
   
  * Typical range for text data: 0.7-0.99

**4. Jarque-Bera Test (p=0.0):**

* Interpretation: Absolutely non-normal distribution

* Significance:

  * Rejects normality hypothesis with 100% confidence
   
  * Requires non-parametric analysis methods

**5. Hapax Legomena (19,834):**

* Interpretation: Very high count of rare words

* Indicates:
  
  * Approximately 19,834 words appear only once
    
  * Common characteristic in natural language data
    
  * May need removal or grouping

**6. Dis Legomena (9,131):**

* Interpretation: Words with minimal repetition

* Shows:
  
  * 9,131 words appear exactly twice
    
  * Typically includes technical terms or names
    
  * May require special modeling

**7. High-Frequency Tokens (>100 Occurrences: 10,259):**

* Interpretation: Very common words

* Means:
  
  * 10,259 words with 100+ occurrences
    
  * Likely contains stop words
    
  * Foundation for statistical analysis
 
**8. Top 10% Threshold (508.4):**

* Interpretation: Boundary between frequent/rare tokens

* Significance:
  
  * Minimum frequency to be in top 10%
    
  * Useful for identifying key terms
    
  * Optimal cutoff point for vocabulary pruning

**9. Interquartile Range (IQR: 23.0):**

* Interpretation: Middle 50% token spread

* Indicates:

  * Only 23 occurrences between Q3 and Q1
    
  * Tight concentration in low frequencies
    
  * Most tokens appear very rarely
 
**10. Distribution Deciles:**

* Interpretation: Frequency cut points

* Key Values:

  * 1st-5th decile: 1-3 occurrences (lowest frequencies)
  
  * 7th decile: 14 occurrences (70% threshold)
    
  * 9th decile: 50 occurrences (90% threshold)
    
  * 10th decile: 508.4 occurrences (top 10%)
 
**11. Top/Bottom 1% Ratio (62,408.16):**

* Interpretation: Extreme frequency gap

* Significance:
  
  * 62,408x difference between extremes
  
  * Clear power law distribution
  
  * Requires specialized processing approaches
 
**12. Top 10% Tokens Share (99.68%):**

* Interpretation: Extreme concentration in few tokens

* Indicates:
  
  * 99.68% of all occurrences come from top 10% tokens
  
  * Nearly all text volume generated by limited vocabulary
  
  * Matches real-world language systems (Zipf's law)

**13. Top 100 Tokens Share (66.09%):**

* Interpretation: Dominance of high-frequency tokens

* Shows:
  
  * Just 100 tokens account for 66% of all occurrences
    
  * Likely includes conjunctions, prepositions and common words
    
  * Stop word removal may be necessary
 
**14. Herfindahl Index (0.017):**

* Interpretation: Moderate vocabulary concentration

* Scale:
  
  * 0 = Perfect equality
    
  * 1 = Complete monopoly (Current: 0.017)
    
  * Suggests several ultra-frequent tokens
 
**15. Normality Test (p=0.0000):**

* Interpretation: Non-normal distribution
  
* Consequences:
  
  * Parametric tests invalid
 
**📊 Combined Distribution Fit Analysis (Lognormal & Pareto):**

* Test Results:

  * Lognormal Fit: p = 0.0 ✗ (Rejected)
    
  * Pareto Fit: p = 0.0 ✗ (Rejected)
    
* Key Interpretations:
  
    * 1.Lognormal Rejection → Your data is more skewed than lognormal can model

      * Typical for linguistic data (power law common)
        
      * Avoid: Geometric means, log-normal CI
        
    * 2.Pareto Rejection → Your tails are heavier than standard Pareto.

      * Implies extreme token dominance (e.g., top 0.1% tokens control >90% frequency)
