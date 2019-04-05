import model

def main():
    filter_keywords = ['smoking', 'tobacco', 'cigarette', 'cigar', 'hookah', 'hooka']
    lda_model = model.main(filter_keywords)

main()