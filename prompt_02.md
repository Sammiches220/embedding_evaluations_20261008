Attached is a text file that has aggregated files for a Python project.

The idea of the project is to evaluate embeddings through various methods and provide a human readable report. 

The report is exactly how I want it formatted and I love the idea of providing same-ordered features for efficiency.

However, I do not like how confusing it is to prepare the data. There was an idea of having a prompt to prep the data for you or generate a script to analyze it? But it is messy and confusing.

Say I have a parquet dataset that has a bunch of feature columns that may not be the disjoint labelled clusters that the original project was optimized for. I would like the project to make it clear what is desired. The prompt idea is fine. But remove the Preparation notes and validation and anything that is unnecessary. The problem I encountered is that there are many files to consider. Just make it simple. The configs are fine, I like those for the most part. The problem I encountered is that there were tons of data type isues when I had an LLM try to prep some data. I don't know what the project really expects or how to provide the information. Especially I'm not sure if the dataset I have with engineered features can be translated without information loss. Like, are we to provide the increasing label separated datasets or what? 

Please try to also modularize the project for navigation ability. This also means making it simpler, and specifically making the documentation much more modular in that the top exposed docs are very brief while the deeper ones are more in depth with a doc that goes over the concept.

Please do consider efficiency. The increasing labels with same dimensions matching rows was for space and time complexity. Take similar approaches still (i.e. don't change this if it's already good. Importantly the disjoint cluster format should stay the same, but there should be some helping information or something to format normal cluster feature columns to the format desired and such)

I want you to please completely reprovide the project. DO NOT CHANGE WHAT I SAID I LIKED. DO NOT SUPPORT OLD LEGACY FORMATS.