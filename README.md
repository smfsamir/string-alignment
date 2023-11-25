# string-alignment
A no-frills package for computing statistically informed alignments between discrete text sequences. 

## Algorithms

### Pointwise mutual information
A very basic alignment algorithm. It will return a matrix containing the pointwise mutual information (PMI) between all of the symbols in the alphabet. 

Advantages:
1. Pools information across datapoints for computing the pointwise mutual-informations. Contrast this against e.g., Needleman-Wunsch, which does not aggregate information across datapoints. 

Disadvantages:
1. It is symmetric, since the PMI is symmetric -- PMI(a,b) == PMI(b,a).
2. It treats sequences like a bag of words -- no positional information.
