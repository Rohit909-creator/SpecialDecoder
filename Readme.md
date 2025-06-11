## This is an experiment on LLMs, based on recent findings from mechanistic Interpretability

Btw, this code is a continuition of my SimpleLLM repo,

So based on a recent video I saw, LLMs need deep layers to make complex circuits and get complex understandings,
so for that we need large number of the LLM-Blocks, so how about this, what if could make a specific layer or a new kind of block to model what the other layers do, basically said,

if we actually had,

embeddings -> block1 -> block2 -> block3 -> block4 -> block5 -> block6 -> ................... blockN -> lmhead

what if could do,

embeddings -> block1 -> block2 -> block3 -> block4 -> SpecialBlock -> blockN

thus replacing the computation of blocks from 5 to N-1 with SpecialBlock,

Where Special Block can be a a recurrent transformer model or someother model which does computation (N-1 - 6) => (N-7) times to model deep representations.

I have to do the rest of the work, feeling too sleepy now, so baaki tommorow