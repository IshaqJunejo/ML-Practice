## Day 15 - Recurrent Neural Networks

Learned about the theory behind `Recurrent Neural Networks`, and worked on implementation of `next character prediction` on `Tiny-Shakespeare` dataset.

Trained 2 models for the same purpose to compare the performance difference of `SimpleRNN` and `GRU` **(Gated Recurrent Unit)** in processing sequential data.
The results that I got from both the models were *Dumb* to say the least, it likely was because of working on a small model, not having enough layers, or maybe not training for enough epochs.

Anyways, here are the results.
```
Seed String: "What is it that you came here lookin"
Generating next "20" Characters

Result based on Recurrent Neural Network
What is it that you came here lookinBBrsBpBppBpBBBBBpsBB

Result based on Gated Recurrent Unit
What is it that you came here lookinGzeSdOxs.!lmOhpI Umj
```

Results from both the models are quite bad (so was their performance in terms of accuracy and loss during training), but the `GRU` have somewhat outperformed `SimpleRNN` as it is not simple repeating the same characters again.

**Next up**: Trying the same task on `LSTM` (Long Short Term Memory).

*Side-Note: Even though the last "Day" of practice was a month ago, I am still going to name it "Day-16". Because, the "Days" don't actually represent the sequence relative to the calender, they represent the sequence relative to my personal Machine Learning practice.*
