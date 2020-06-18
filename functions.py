def normal(s):
    return s

def normal_exp(s):
    nor = []
    for i in range(len(s) - 1):
        try:
            if s[i+1] * s[i] != 0: nor.append(log(abs(s[i+1]), abs(s[i])))
            else: nor.append(0)
        except Exception as e:
            return normal_lin(s)
    return nor

def normal_squared(s):
    return [i ** 2 for i in s]

def sum_sdev(l):
    m = sum(l)/len(l)
    sdev = (sum([(p - m)**2 for p in l])/(len(l)-1))**.5
    return sdev

def normal_lin(s):
    nor = []
    for i in range(len(s) - 1):
        if s[i] != 0: nor.append(1 + (s[i+1] - s[i])/s[i])
        else: nor.append(0)
    return nor

def normal_ari(s):
    nor = []
    for i in range(len(s) - 1):
        nor.append(s[i+1] - s[i])
    return nor

def normal_ari_squared(s):
    nor = []
    for i in range(len(s) - 1):
        nor.append((s[i+1] - s[i])**2)
    return nor

def normal_sdev(l):
    nor = []
    m = sum(l)/len(l)
    sdev = (sum([(p - m)**2 for p in l])/(len(l)-1))**.5
    for p in l:
        d = (p - m)/sdev
        nor.append(d)
    return nor

def normal_sdev_squared(l):
    nor = []
    m = sum(l)/len(l)
    sdev = (sum([(p - m)**2 for p in l])/(len(l)-1))**.5
    for p in l:
        d = ((p - m)/sdev) ** 2
        nor.append(d)
    return nor

def sum_ari(s):
    return sum(s)

def sum_squared(s):
    sm = 0
    for i in s:
        sm += i ** 2
    return sm 

def sum_lin(s):
    r = 1
    for p in s:
        r *= p
    return r

def sum_exp(s):
    r = s[0]
    for p in range(len(s[1:])):
        r ** s[p]
    return r

def sum_max(s):
    if len(s) == 0: return 0
    return max(s)

def sum_min(s):
    if len(s) == 0: return 0
    return min(s)

def sum_avg(s):
    if len(s) > 0:
        return sum(s)/len(s)
    else:
        return 0

def sum_range(s):
    return (max(s) + min(s))/len(s)

def sum_moving_avg(s):
    sm = 0
    for i in range(len(s)-1):
        sm += (s[i+1]+s[i])
    return sm/len(s)
