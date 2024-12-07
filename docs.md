# LuaLuaLua report
 
## Task Description


## Implementation
We decided to base our solution on a pretrained model called GloVe. GloVe, which stands for Global Vectors for Word Representation, is a popular word embedding technique that captures semantic relationships between words in a vector space. This model is a straightforward word-vectorization model, which uses a 50-dimensional vector for each word. In this model (as in other word-vector models), each word is represented by an n-dimensional vector. The "distance" between vectors represents the similarity between words. Distance is calculated using the Pythagorean theorem, as the square root of the sum of squared differences. For instance, the distance between words like "mug" and "cup" is small, as they are similar and can often be used interchangeably in a sentence. However, words like "tractor" and "skyscraper" are positioned far apart in the vector space due to their distinct meanings.

There are, of course, other word-to-vector models, but we chose GloVe for our initial attempt due to its simplicity and ease of use and we found it good enough for our current purposes, so we got stuck with it. 

### Data processing
To use GloVe we needed to encode our text input into word-vectors, but first the tweets needed cleaning. 
1. **Cleaning the Text**:  
   - Removing *stopwords* (e.g., "of," "for," "with," "haven't," "by"). These are common words that usually add little value to the analysis.  
   - Eliminating non-letter characters, including numbers.  
   - Replacing links with a `[link]` placeholder.  

   Surprisingly, `[link]` became the most frequent "word" in our dataset, far outpacing any other term.  

   ![Distribution of words in training data](https://github.com/user-attachments/assets/7b4ba618-bc03-4d00-bac2-80d1e9108e72)  
   *Distribution of words in training data*  

   The abundance of `[link]` is likely due to the large number of tweets that included images or other shared content. Since links don't convey much meaning, we decided to remove them entirely from the dataset.  
   Actually, after completing the model, we were courious about the fact that how the cleaning affects the output of the model, since despite the fact, that these words do not carry information, the connection between the words might be useful. So we just simply removed the cleaning commands form the code. Here is the comparsion of the confusion matrix between the models with and without the text cleaning:
   ![Without the cleaning](image.png)
   
   As it can be seen, cleaning does not affect the out very much, but its got a slightly better performance, so we just kept the orginal form.
2. **Addressing Retweets and Usernames**:  
   After removing links, we noticed a significant number of meaningless words remaining in our dataset. Among the most common was `RT`, which we soon realized indicates retweets. Additionally, many other frequent "words" appeared to be fragments of usernames from retweet headers (e.g., `@user1234`). These usernames commonly don't carry much important information, so we removed them together with the retweet headers.

3. **Tokenizing the tweets**:
   After cleaning we tokenized each tweet into a list of words. We plotted the distribution of tweet lengths.
   
   ![image](https://github.com/user-attachments/assets/47921616-bd9e-44d6-93d7-578cc8bb5c63)
   
   *Distribution of number of tokens per cleaned tweet*
   
   We needed all inputs for our model to be the same in length. 99% of the words tweets is 17 or less words long so we truncated the longer tweets and padded the shorter ones.

### Embedding matrix



   
## Evolution of the Model
Basic architecture

### 1. LSTM 
**Model:** The first model (whitch we created to the II. Milestone) used LSTM (Long Short-Term Memory) architecture, but it was just kind of a dummy.
**Evalutaion:**
The result were quite poor, but the task for the II. Milestone was to prepare everything in our notebook except the model, so it did the job. The confusion matrix looked like this:
![alt text](image-2.png)


### 2. Improved LSTM
**Model:**
**Evalutaion:**

### 3. Two phase learning
**Model:**
**Evalutaion:**

### 4. Stacked Model: GRU-LSTM
**Model:**
**Evalutaion:**
![Final confusion matrix](image-1.png)

## Conclusion
BERT might have been better...

## About AI usage
