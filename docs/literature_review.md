Last update: 10 Feb 2020

Overview

Most improvements and ideas presented in the following
papers have been incorporated to BERT already such as
contextual embeddings and multi head attention. That
said there some ideas that can help us drive our BERT
model further such as:
* Journal, author and other embeddings (MeshProbeNet)
* Train using more text than when testing/predicting
* Enseble of TFIDF and BERT models (DeepMesh)
* Train model to predict number of labels

At the same time, it noteworthy that our approach has
ignored tree based alternatives like SLEEC, the existence
of some biomedical and other dataset that we could
test our approach like Wiki-500 and HoC and finally
the correlation among labels which is partly modelled.

# DeepMesh
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908368/

DeepMesh is the best approach for Mesh indexing that has won
most BioASQ competitions. The approach is involving from TFIDF
based to incorporating document embeddings to attention. This
paper describes the approach before the authors incorporated
attention in the last competition.

The approach consists of three steps, a) producing Mesh candidates
b) ranking them and c) calculating number of mesh tags. Producing
Mesh candidates is based on “global” and “local” evidence by which
the authors mean they either classify agnostically using a document
representation (global) or retrieve similar documents and their tags
(local). The global approach is based on binary relevance and SVM
while the local one is based on KNN. The authors experiment with
a TFIDF representation as a D2V separately but also together. To rank
and decide the number of tags they use the scores produced by
both the global and local approach as well as the frequency of a tag
and the number of tags from the neighbouring documents.

![Model architecture](https://user-images.githubusercontent.com/4975761/74143436-7b91ff80-4c03-11ea-90d9-f5d168e15de7.png)

The authors find that TFIDF performs much better (+10%) alone than
D2V alone but when combined together they get an additional 5%.
Looking further into the reasons for this they find that TFIDF performs
better on tags with fewer data while D2V performs better on tags with
more. This was at first surprising given that a dense representation should
perform well in cases where not a lot of data is available but it makes
sense given that the vectors are trained on less data in those cases
and thus learn less good representations. As with previous approaches
learning to rank has been shown as an effective way to combine different
evidence. Finally DeepMesh was shown to perform 2% better than the
previous best MeshLabeler from the same authors which did not use
embeddings.

# X-BERT
https://arxiv.org/abs/1905.02331

The authors are inspired from information retrieval approaches and the recent
improvement in NLP i.e. BERT and they propose a three step process to
combine the two as a means to make BERT relevant to the extreme multi-label
classification problem. The labels are treated as “documents” in the information
retrieval setting and the text as the “query”. The approach contained three steps,
indexing, matching and ranking to produce the final labels.

The first step focuses on representing the labels semantically which is achieved
using ELMO and TFIDF weights of the label text and documents that are associated
with a label. These representations are clustered into K label groups using K means
to reduce the number of labels and make the problem more feasible.

The second step trains a neural network to map from the query text to the label groups.
A label group is considered positive if any of the labels is associated with the text.
This is also a multi label classification problem with the difference that the number of labels
is controlled by the authors since it is equal to the number of clusters. The experiment
with biLSTM and self attention as well as BERT for this step.

The final step is about assigning a probability to every label from the label groups that
have been retrieved. The authors use a one vs all linear classifier but also ensemble from
different runs of BERT and clusters (this step is a bit opaque but the general idea is clear).

The authors report improvements of about 5% from the second best method but also
mention that different papers report different numbers for some models and direct
comparison is not always possible due to different preprocessing. They perform some
ablation studies showing the superiority of BERT vs self attention and of ranking model
vs simple TFIDF. The biggest improvement seems to come from using a neural network
approach for the second step instead of a linear model.

# AttentionXML
https://arxiv.org/abs/1811.01727

The authors propose a combined tree based and attention based architecture.
A shallow and wide PLT is build and for each level an attention based neural
network is being trained. The architecture is a biLSTM on top of Glove
pre trained embeddings that is followed by a multi label self attention layer
which means that for each output label a different set of attention weights
produce a representation relevant to the label which is then fed to a
a fully connected layer and a sigmoid. Binary cross entropy loss
is used as a loss function.

![Model architecture](https://user-images.githubusercontent.com/4975761/73938767-99542180-48f0-11ea-966c-11e61958c1b1.png)

The authors evaluate the model using Precision@k and achieve state of the
art results for a number of big multi label datasets such as wiki-500 and
Amazon-3M. Their approach performs on average 5% better from the second
best. The authors also perform an ablation study in which they show that
the attention layer is responsible for much of the performance difference
compared to a vanilla BiLSTM or even XML CNN. Interestingly BiLSTM
outperforms XML CNN on their comparison. They also use stochastic
weight averaging and also show that it adds an additional 2% increase.
Finally they show that performance on tail labels is significantly improved,
something that is the result of using trees more than attention since
weights are only shared for the first biLSTM part.

# MeshProbeNet
https://academic.oup.com/bioinformatics/article-abstract/35/19/3794/5372674?redirectedFrom=fulltext

The most common approaches for indexing new pubmed articles with MeSH headings
are [MTI](https://ii.nlm.nih.gov/MTI/), [MetaMap](https://metamap.nlm.nih.gov/), MetaLabeler, MeshLabeler and DeepMesh. MTI is the tool developed
by NLM that relies on the MetaMap tags of similar pubmed articles. MetaLabeler trains one
binary classifier per mesh term whereas MeshLabeler aims to improve MetaLabeler by
incorporating additional data such as similar publications and term frequencies. DeepMesh
builds on top of MeshLabeler and adds deep representations in the form of word embeddings.

There a number of shortcoming of these methods that the authors are trying to address. Firstly,
previous approaches do not take advantage of the sequence nature of the input data, then
training one model for each mesh term is computationally expensive and most importantly
does not take into account the correlation between mesh terms. Lastly, these approaches
rely on retrieving similar articles from a database which is slow and not memory efficient. To
address this, the authors design a unified neural network consisting of bi GRU units,
[self attentive units](https://arxiv.org/abs/1703.03130) (probes) and fully connected layers.

The input of the neural network are word embeddings which are fed into a bidirectional
Gated recurring unit layer that better models the sequence nature of the data. The hidden states
of each word in combination with n probes produce the attention weights that are used
to calculate n context vectors which are concatenated and fed into a three layer neural
network which is able to model correlation among terms in its weights before ending
up in a sigmoid. Finally sharing of the weights allows tags with few data to still perform well.

![Model architecture](https://user-images.githubusercontent.com/4975761/73927849-d31a2d80-48da-11ea-8ae9-44add08887c0.png)

The authors achieve state of the art results in the 2018 [BioASQ challenge](http://bioasq.org/) outperforming
DeepMesh by 2% in most measures. They also show that the self attentive probes learn to attend
for different biomedical relevant information such as diseases. Additionally they perform
an ablation study in which they demonstrate that this layer adds approximately 3% to
the performance compared to a vanilla GRU and fully connected NN which is significant.

# Transfer Learning in Biomedical Natural Language Processing: An Evaluation of BERT and ELMo on Ten Benchmarking Dataset
## Peng, Yifan, Yan, Shankai, Lu, Zhiyong
https://arxiv.org/abs/1906.05474

The authors combine ten existing dataset into a benchmark evaluation
they name BLUE inspired by GLUE so that different transfer learning
approaches can be compared on the same data. The datasets consist of
10 datasets and five different tasks, two sentence similarity tasks, three
named entity recognition, three relation extraction, one document classification
and one inference. The latter is a multilabel document classification
which they classify sentence by sentence and combine the labels produced.

They compare ELMO and BERT as one of the most popular transfer learning techniques
nn the dataset by pre training on pubmed abstracts and clinical documents. BERT outperforms
ELMO by 20 points in sentence similarity and around 3 points in other tasks. BERT
base performs better than BERT large in most datasets and only outperforms data
in data with longer sentences which is an interesting observation. Finally pre training
on pubmed only gives very strong performance and the clinical data offer little
improvement.

# Deep Learning for Extreme Multi-label Text Classification
## Liu, Jingzhou, Chang, Wei-Cheng, Wu, Yuexin, Yang, Yiming
http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf

The authors are motivated by the lack of neural network specific architecture for
extreme multi label classification. The most common non deep learning approach
is target embedding such as [SLEEC](https://papers.nips.cc/paper/5969-sparse-local-embeddings-for-extreme-multi-label-classification) and tree based ensemble methods such as
FastXML. The most common deep learning approaches are Bow CNN, [FastText](https://fasttext.cc/),
[CNN Kim](https://arxiv.org/abs/1408.5882) etc.

The authors extend CNN Kim by adding dynamic max pooling, a bottleneck layer
and a suitable loss function which they find all add to the performance improvement
via an ablation study. Dynamic max pooling translates to generating p features per
filter instead of 1 by applying the max operation to p chunks of the filter output.
The loss found to perform better is binary cross entropy. Finally the hidden bottleneck
layer compresses the representation of the document to a vector that better represents
the document than the output from the pooling.

![Model architecture](https://user-images.githubusercontent.com/4975761/73925846-546fc100-48d7-11ea-8828-85957b7b5e82.png)

The authors test their approach on 6 datasets that contain from 103 labels (RCV1) to
Wiki 500k and perform significantly stronger in datasets that contain more data which
indicates that their approach scales well and takes advantage of more data when available.
At the same time, results how that a strong baseline can be achieved by just using SLEEC
which is a not a deep learning method. SLEEC is at most 2% points behind the best method
and the authors observe that it outperforms neural network methods for longer documents.
Note that FastText and CNNBow, which are not designed for a multi label scenario thus
they do not model the correlation among labels well suffer greatly as the number of label
increases. This point is reinforced by looking at a subset of the datasets in which authors
find that only FastXML and CNNXML scale well with the number of labels.

# ML-Net, multi-label classification of biomedical texts with deep neural networks
## Du, Jingcheng, Chen, Qingyu, Peng, Yifan, Xiang, Yang, Tao, Cui, Lu, Zhiyong
https://arxiv.org/abs/1811.05475

This paper was published prior BERT and investigates an end to end approach to Multi label text classification.
It applies Hierarchical attention networks to the problem which were introduced by Microsoft research as a
means to capture the hierarchical nature of text i.e. words combine to sentences and sentences to documents.
Hierarchical attention networks combine the words of a sentence using a BiLSTM layer and one attention layer 
into a sentence embeddings. These embeddings are similarly combined to document via a similar layer.
On top of the hierarchical attention network which produces document embeddings the author train two neural
networks, one that predicts the number of labels for a document and the other that scores how likely a label is.
This novel architecture aims to remove the need of having a threshold that needs to be adjusted.

![Model architecture](https://user-images.githubusercontent.com/4975761/73938687-73c71800-48f0-11ea-9067-d1dec411330e.png)

They test their approach to three datasets one of which is ICD and the other being less than 50 labels in total, 
one is cancer classification, the other is chemical exposure.
They achieve a 10%, 1% and 3% boost in f1 from the strong baseline of tiff-svm. Interestingly the get the most
boost in the dataset with the few examples.

# Data and text mining BioBERT: a pre-trained biomedical language representation model for biomedical text mining
## Lee, Jinhyuk, Yoon, Wonjin, Kim, Sungdong, Kim, Donghyeon, Kim, Sunkyu, So, Chan Ho, Kang, Jaewoo
https://arxiv.org/abs/1901.08746

BioBERT is BERT trained on PubMED data. The data is a combination of abstracts and full texts when available.
It took 23 days on 8 V100 GPUs to train BioBERT something that on AWS would cost approx 10K. The authors
used the same architecture as BERT and same fine tune methodology.

The authors test BioBERT to three separate groups of datasets, 9 named entity recognition related ones such NCBI disease,
3 relation extraction ones such as CHEMPROT and in biomedical QA on data from BioASQ task b. The achieve near state of the
art results in most datasets with BioBERT outperforming BERT typically from 2% to 4% approx indicating that pre training on
biomedical text is beneficial to biomedical problems. They also find that up to 1B words performance increases significantly,
then it plateaus.

# Results of the seventh edition of the BioASQ Challenge (not published yet)
## Nentidis, Anastasios, Bougatiotis, Konstantinos, Krithara, Anastasia, Paliouras, Georgios

BioASQ is a competition that is running since 2012 with two tasks. The first task is biomedical indexing of pubmed
articles with Mesh tags. The second task is a biomedical QA task that is broken into two phases a) finding the relevant
snippets from pubmed articles and b) extracting the exact answer from the snippet.

In regards to the first task which is the most relevant to science tagging, the competition takes advantage of the fact that
NLM manually tags pubmed articles with a delay which allows researchers to predict the tags using machine learning in
the meantime and then compare the results with the manual annotations after they become available. As training set 14M
priorly tagged documents are provided with around 100K new ones tagged during the competition.

12 teams participated and once more DeepMesh proved to be the most effective approach. The metric used is lowest common
ancestor f measure that takes into account the hierarchical nature of Mesh. This year the system consistently performed above 70%
with the MTI baseline (the automatic system that NLM uses) around 65%. There was a big jump in improvement this year possibly
due to BERT as performance had plateaued in last two competitions in around 66%.

DeepMesh uses doc2vec, tfidf embeddings as well as the mesh label system and learning to rank which are described in separate papers.
This year the authors incorporated AttentionXML which is again work described on a different paper.

Another interesting approach from NLP uses a CNN to improve MTI combining text embeddings with journal information and year, the 
last to account for concept drift over the years.

Todo
* Add table of indicative results from the paper.
* Add link to code
