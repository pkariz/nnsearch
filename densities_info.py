from nnsearch.datasets.density import density
from nnsearch.datasets import samples, load_dataset
import time

ps = [-1]
text = ""
t, a, b = 50, 8, 64
start_all = time.time()
for p in ps:
    text += "------------------p:%f-----------------\n" % (p,)
    print "------------------p:%f-----------------" % (p,)
    for x in sorted(samples):
        dataset = load_dataset(x)
        dataset.normalize()
        start = time.time()
        mean, var = density(dataset.data, p=p, t=t, a=a, b=b)
        elapsed = time.time() - start
        text += "Dataset:%s, density --> mean:%f, var:%f, time needed:%f\n" % (x, mean, var, elapsed)
        print "Dataset:%s, density --> mean:%f, var:%f, time needed:%f" % (x, mean, var, elapsed)
elapsed_all = time.time() - start_all
text += "time needed:%f" % (elapsed_all,)
#filename = "density_info_exact_"+str(t)+"_"+str(a)+"_"+str(b)+"_2.txt"
#with open(filename, "w") as f:
    #f.write(text)

