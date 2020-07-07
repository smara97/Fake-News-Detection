# Fake News Detection

- Help know news are fake or real , measure quality of article and sort the authors by the quality and reality of article .
- Get ~63% Accuracy using Liar plus data-set, This state of the art  model without using (Attentions layers) BERT model .



## Fake News

Depended on 3 featuers of data to training : Statement , Subject and Justification in Liar Liar plus dataset 


### Liar plus :

This dataset has evidence sentences extracted automatically from the full-text verdict report written by journalists in Politifact. Our objective is to provide a benchmark for evidence retrieval and show empirically that including evidence information in any automatic fake news detection method (regardless of features or classifier) always results in superior performance to any method lacking such information.

Below is the description of the TSV file taken as is from the original [LIAR dataset](https://www.aclweb.org/anthology/W18-5513/), which was published in this [paper](https://www.aclweb.org/anthology/P17-2067/). We added a new column at the end that has the extracted justification.

- Column 1: the ID of the statement ([ID].json).
- Column 2: the label.
- Column 3: the statement.
- Column 4: the subject(s).
- Column 5: the speaker.
- Column 6: the speaker's job title.
- Column 7: the state info.
- Column 8: the party affiliation.
- Columns 9-13: the total credit history count, including the current statement.
- 9: barely true counts.
- 10: false counts.
- 11: half true counts.
- 12: mostly true counts.
- 13: pants on fire counts.
- Column 14: the context (venue / location of the speech or statement).
- Column 15: the extracted justification.

The dataset can be found in this [commit](https://github.com/smara97/FakeNews/tree/master/liar-plus).

Using LSTM in trained model and glove file to word embeddings .

After transfer each word to vector of numbers has size 300 dims ,ex:
'trumb' --> [1,4,.,.,.,.,.,.,.,.,.,9] , then each sentence have vectors of vectors numbers.

Using mathematics trick :
get cosine similarity between mean of each featuers and mean of another featuers.

mean(tarnsferd_statement) + cosine similarity between mean(tarnsferd_statement) and mean(tarnsferd_subject) + mean(tarnsferd_subject) + cosine similarity between mean(tarnsferd_subject) and mean(tarnsferd_justification) and mean(tarnsferd_justification) + cosine similarity between mean(tarnsferd_justification) and mean(tarnsferd_statement) .


25 epochs to training model , if you need model learn more increase the number of epochs .

more +3 versions of project [the last version of project](https://github.com/smara97/FakeNews/blob/master/liarplus_version_1.ipynb)

example of new:

statement="The decade that shattered trust in politics."

subject="politics."

justification="It is totally normal for ministers and officials in high pressure jobs to have quarrels and tricky conversations.
Arguably, a bit of healthy tension can be a good thing for governments, to make sure that ideas are tested and policies properly thought through.
It is also normal from time to time for senior officials to move quietly to different government departments if a relationship breaks down with their political boss, or sometimes, for them to retire early if the situation has become impossible.
There is nothing remotely normal however about a top government official quitting their job, suing the government in the belief they were forced out, deciding to go public with the reasons, and accusing one of the most senior politicians in the country of not being straight with the truth.
But that is exactly what's happened. Sir Philip Rutnam has been one of the most senior civil servants for years, in charge at the Home Office for the last few.
His time there has not always been an unalloyed success - the Home Office, as one of the biggest and most complicated departments in the government, has struggled with various issues, most notably the Windrush scandal. The Home Office is often seen as a poisoned chalice given the nature of its job."


the output : The credibility of new: 51.84%


## Quality of text 

Using Blue Score model to test the dependency of sentence by togther 
[exist one version](https://github.com/smara97/FakeNews/blob/master/quality_text_version1_.ipynb)


After get the credibility and quality of each new,can sort them from most credibility and quality.  

