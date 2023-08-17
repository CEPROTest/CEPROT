import math
from collections import Counter

class GLEU :

    def __init__(self,n=4,l=1) :
        self.order = 4
        self.weight = l

    def load_sources(self,spath) :
        self.all_s_ngrams = [ [ self.get_ngram_counts(line.split(),n) \
                                  for n in range(1,self.order+1) ] \
                                for line in open(spath) ]

    def load_references(self,rpaths) :
        refs = [ [] for i in range(len(self.all_s_ngrams)) ]
        self.rlens = [ [] for i in range(len(self.all_s_ngrams)) ]
        for rpath in rpaths :
            for i,line in enumerate(open(rpath)) :
                refs[i].append(line.split())
                self.rlens[i].append(len(line.split()))

        self.all_r_ngrams = [ ]
        for refset in refs :
            all_ngrams = []
            self.all_r_ngrams.append(all_ngrams)

            for n in range(1,self.order+1) :
                ngrams = self.get_ngram_counts(refset[0],n)
                all_ngrams.append(ngrams)
                for ref in refset[1:] :
                    new_ngrams = self.get_ngram_counts(ref,n)
                    for nn in new_ngrams.elements() :
                        if new_ngrams[nn] > ngrams.get(nn,0) :
                            ngrams[nn] = new_ngrams[nn]


    def get_ngram_counts(self,sentence,n) :
        return Counter([tuple(sentence[i:i+n]) for i in xrange(len(sentence)+1-n)])

    def set_lambda(self,l) :
        self.weight = l

    def gleu_stats(self,hypothesis, i):

      hlen=len(hypothesis)
      rlen = self.rlens[i][0]

      for r in self.rlens[i][1:] :
          if abs(r - hlen) < abs(rlen - hlen) :
              rlen = r

      yield rlen
      yield hlen

      for n in xrange(1,self.order+1):
        h_ngrams = self.get_ngram_counts(hypothesis,n)
        s_ngrams = self.all_s_ngrams[i][n-1]
        r_ngrams = self.all_r_ngrams[i][n-1]

        r_ngram_diff = r_ngrams - s_ngrams
        for k in r_ngram_diff.keys() :
            if k in s_ngrams :
                del r_ngram_diff[k]
        s_ngram_diff = s_ngrams - r_ngrams
        for k in s_ngram_diff.keys() :
            if k in r_ngrams :
                del s_ngram_diff[k]

        yield sum( (h_ngrams & r_ngram_diff).values() ) + \
            max([ sum( (h_ngrams & r_ngrams).values() ) - \
                  self.weight * sum( (h_ngrams & s_ngram_diff).values() ), 0 ])

        yield sum( (h_ngrams & r_ngram_diff).values() ) + max([hlen+1-n, 0])



    def gleu(self,stats):
       if len(filter(lambda x: x==0, stats)) > 0:
         return 0
       (c, r) = stats[:2]
       log_gleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.

       return math.exp(min([0, 1-float(r)/c]) + log_gleu_prec)
