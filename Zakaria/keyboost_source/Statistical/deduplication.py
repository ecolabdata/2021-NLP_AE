def deduplication(key_rank,n_top,transformers_model,tresh=0.9):
  model = select_backend(transformers_model)

  keywords = [(key_rank.values[0][0],key_rank.values[0][1])]

  for row in key_rank.values:
    skip=False
    for k in keywords:
      sim = cosine_similarity([model.embed(row[0])],[model.embed(k[0])])[0][0]
      if sim > tresh:
        skip=True
        break
    if not skip:
      keywords.append((row[0],row[1]))

  return keywords if len(keywords)< n_top else keywords[:n_top]
