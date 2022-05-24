from ContrastSystem import ContrastEnhancemer as CE

if __name__ == "__main__":
    ce = CE()
    ce.Init_Input()
    ce.InitRules()
    for i in range(256):
        val = ce.compute(i)
        print("I: ",i,"Value: ",val)