## First, get precise about what a "mask" is

This matters because it determines what you're actually measuring.

**Wanda** scores each weight as $|W_{ij}| \cdot \|X_j\|_2$, comparing within each output row. The only calibration-dependent term is $\|X_j\|_2$, the L2 norm of input channel $j$ over calibration tokens. So the entire "language signature" of a Wanda mask is one vector per linear layer, of size $d_{in}$. That's a few MB for a whole 8B model.

Two consequences. You can store activation norms instead of masks and regenerate masks on demand, which saves you from writing 144 bitmaps at ~875 MB each. And you can measure transfer at the *score* level (cosine similarity or Spearman rank correlation between norm vectors) rather than only on thresholded binary masks, which gives you a continuous signal with no threshold artifacts.

**SparseGPT** entangles mask selection with weight reconstruction through the Hessian $XX^\top$. That's not a problem, it's a bonus factorial cell: mask from language A with reconstruction from language B, and vice versa. If reconstruction data matters more than mask choice, that's a publishable finding on its own.

**Magnitude pruning** is calibration-free, so its cross-language mask overlap is 1.0 by construction. Useful as a quality baseline, useless as a transfer datapoint.

## The trap that will sink the paper if you miss it

Suppose you find IoU of 0.93 between Tamil and Telugu masks. That number means nothing in isolation, because you need both ends of the scale.

For two independent random masks at keep-fraction $k$, expected IoU is $k/(2-k)$. At 50% sparsity that's 0.333. At 70% sparsity, $k=0.3$, so 0.176.

The upper end is the within-language noise floor: two masks from *different random calibration samples of the same language*. Run 3 to 5 seeds per language to get this. It'll be high, maybe 0.95.

Then report an adjusted score:

$$\text{transfer} = \frac{\text{IoU}_{\text{cross}} - \text{IoU}_{\text{random}}}{\text{IoU}_{\text{within}} - \text{IoU}_{\text{within-seed-noise}}}$$

With 0.93 cross, 0.95 within, 0.333 random, adjusted transfer is about 0.97. Nearly everything that *could* vary by language doesn't. Without the baselines you'd have written either "masks are 93% similar, transfer is fine" or "7% of the mask differs, language matters," and both would be unfounded.

## Controlling the content confound

Calibrating Hindi on Hindi Wikipedia and Tamil on Tamil Wikipedia means the topics differ, so any mask difference could be domain rather than language. Use a parallel corpus so the semantic content is held fixed: FLORES-200 covers all the languages you need with identical sentences.

Then handle fertility, which bites here directly. Wanda's default is 128 sequences of 2048 tokens. Since Indic text costs 3 to 5 times more tokens per unit of content, a token-matched calibration set gives Hindi far less actual content than English. Run both regimes and report both:

- **Token-matched**: equal calibration compute, unequal content
- **Content-matched**: equal sentences, unequal token counts

If content-matched calibration reduces cross-language differences, then part of what looks like language specificity is just calibration starvation. That's a clean, useful result.

## Language selection as a factorial design

You want to separate script, family, and resource level rather than lumping them into "distance from English."

| Contrast | Languages | Isolates |
|---|---|---|
| Same family, same script | Hindi, Marathi, Nepali | resource level |
| Same family, different scripts | Tamil, Telugu, Kannada, Malayalam | script |
| Different family, similar resource | Bengali vs Telugu | family |
| Same language, two scripts | every language + its romanization | script, cleanly |

That last row is the strongest manipulation and it's available for free. Romanizing holds language, content, and speaker identity fixed while changing script entirely. Konkani (Devanagari and Kannada) and Punjabi (Gurmukhi and Shahmukhi) give you natural-script versions of the same control.

## Where to actually spend compute

Here's my honest prior: at 50% unstructured sparsity, Wanda masks are dominated by the $|W|$ term, overlap will be high, and pooled multilingual calibration will work fine. If you run the whole study at 50% unstructured, you'll get a boring null.

Push into the regimes where the activation term has leverage and headroom is thin:

- **High sparsity**: 60%, 70%, maybe 80%
- **2:4 semi-structured**: hardware-relevant, and the per-group constraint amplifies score differences
- **Structured**: attention head and FFN neuron removal, where you're deleting whole units and language-specific neurons are a real phenomenon

## The evaluation matrix

Build a 12×12 asymmetric matrix: prune with calibration language A, evaluate on language B. Add two extra calibration rows that are the real baselines to beat: English-only (the current default in every pruning paper) and multilingual-pooled.

That pooled row is mandatory. If pooling matches per-language specialization, the deployment question dissolves and the answer is "ship one mask." Include it or a reviewer will, and you'd rather find out yourself.

For the metric, use degradation relative to the dense model in the same language, $\Delta = q_{\text{dense}} - q_{\text{pruned}}$ normalized by $q_{\text{dense}}$. Within-language comparison sidesteps the fact that absolute scores and perplexities aren't comparable across languages with different tokenizations. If you do want raw perplexity, report bits-per-byte instead so fertility doesn't contaminate it.

Cost control: run perplexity or bits-per-byte across the full grid since it's cheap, then downstream tasks (IndicXTREME, MILU, IndicGenBench subsets) at one or two sparsity levels only.

## Turning the matrix into the deployment answer

This is a k-medoids problem on the damage matrix. For $k = 1 \ldots 12$, choose $k$ calibration languages as medoids, assign every language to its best available mask, and record worst-case and mean degradation. Plot $k$ against worst-case degradation. That curve is your headline figure, and it answers the title question with a measured number.

For the clustering story, compare the dendrogram from $1 - \text{IoU}$ distances against script grouping, family grouping, and resource tier, using Adjusted Rand Index or a Mantel test between distance matrices. That tells you *which* factor organizes the space.

## The result I'd most hope for

Report IoU broken down by depth and by projection type (Q/K/V/O, gate/up/down). The language-specific-neuron literature suggests language sensitivity concentrates in early and late layers. If that holds, the engineering answer isn't $N$ full masks. It's one shared mask for the middle two-thirds plus small per-language masks for a handful of layers, which collapses storage cost by an order of magnitude.

That hybrid recipe is a better contribution than the clustering itself, and it only emerges if you log overlap per layer from the start rather than aggregating.

## Minimum viable version

One model (Llama-3.1-8B), 8 languages, Wanda only, 70% unstructured plus 2:4, 3 seeds, FLORES calibration, bits-per-byte plus two downstream tasks. That's roughly 30 to 50 GPU-hours and a complete workshop paper. Expand to Gemma-2-9B and an Indic-centric model like Sarvam-1 for the fuller version, since whether Indic-centric pretraining makes masks more or less language-specific is a genuine open question either way.

Do you want me to sketch the actual code structure for the mask extraction and overlap pipeline, or is the harder question right now which venue and timeline you're targeting?
