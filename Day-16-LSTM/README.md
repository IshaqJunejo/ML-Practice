## Day 16 - Long Short Term Memory

Tried implementing a `Long Short Term Memory` model for the `next character prediction` on `Tiny-Shakespeare` dataset. Just like I previously did with `RNN` and `GRU`.

The results are,
```
Seed String 1: "To be or not to be "
Seed String 2: "Would you proceed especially "
Seed String 3: "What is that you came here for "
Generating next "25" Characters

Results based on Long Short Term Memory Model
1,  to be or not to be the dearest than the comm
2,  would you proceed especially to the death.

clarence:

3,  what is that you came here for the common words.

glouce
```

### Some Takeaways
- `LSTM` are better than `GRU` but maybe not that much. It took me way too much manual Hyperparameter Tuning to come to sensible results.
- Hallucinates almost as much as `GRU` in relatively high temperature sampling, had to use **0.01 Temperature**.
- Takes way to long to be trained properly, which is difficult on a CPU-only machine.
- The results that it is producing are achieved from the checkpoint saved on **Epoch 28** with a validation loss of **1.43610**, after that it started **Overfitting**. I was personally aiming for a validation loss of less than **1.0**.
- **The Most Important Takeaway:** This thing took me longer than a single day (6 days to be exact).
