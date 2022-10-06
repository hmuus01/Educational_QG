import json

# feel free to wrap this into a larger loop for batches 0~99
import os

BATCH_ID = 0

# create a lookup for the pdf parse based on paper ID
paper_id_to_pdf_parse = {}
datapath=os.path.join('data', 'pdf_old')

cs_df = data_factory.get_dev(data)

df1=data_df[['abstract']]
df2 = data_df[['body_text']]
df2=df2.rename(columns={'body_text':'abstract'})
# df3 = pd.merge(df1,df2, right_index=True, left_index=True)
df3 = pd.concat([df1,df2])
df_out = pd.DataFrame(['text'])

with open(f'data/CS/pdf_old/pdf_parses_{BATCH_ID}.jsonl') as f_pdf:
    for line in f_pdf:
        pdf_parse_dict = json.loads(line)
        paper_id_to_pdf_parse[pdf_parse_dict['paper_id']] = pdf_parse_dict
        paper_id = pdf_parse_dict['paper_id']

        print(f"Currently viewing S2ORC paper: {paper_id}")

        # get citation context (paragraphs)!
        if paper_id in paper_id_to_pdf_parse:
            # (1) get the full pdf parse from the previously computed lookup dict
            pdf_parse = paper_id_to_pdf_parse[paper_id]

            # (2) pull out fields we need from the pdf parse, including bibliography & text
            bib_entries = pdf_parse['bib_entries']
            paragraphs = pdf_parse['abstract']

            # (3) loop over paragraphs, grabbing citation contexts
            for paragraph in paragraphs:

                df1 = paragraphs[[]]
                # print(cs_df.head())
                for i, row in tqdm(cs_df.iterrows()):
                    dict_item = row.item()
                    if len(dict_item) < 1:
                        continue
                    text = dict_item[0]['text']
                    entry = pd.DataFrame.from_dict({
                        "text": [text]
                    })

                    df_out = pd.concat([df_out, entry], ignore_index=True)
                df_out.to_csv(os.path.join(datapath, 'cs2.csv'), sep='\t', header=False, index=False)

# filter papers using metadata values
# citation_contexts = []
# with open(f'full/metadata/metadata_{BATCH_ID}.jsonl') as f_meta:
#     for line in f_meta:
#         metadata_dict = json.loads(line)
#         paper_id = metadata_dict['paper_id']
#         print(f"Currently viewing S2ORC paper: {paper_id}")
#
#         # suppose we only care about ACL anthology papers
#         if not metadata_dict['acl_id']:
#             continue
#
#         # and we want only papers with resolved outbound citations
#         if not metadata_dict['has_outbound_citations']:
#             continue
#
#         # get citation context (paragraphs)!
#         if paper_id in paper_id_to_pdf_parse:
#             # (1) get the full pdf parse from the previously computed lookup dict
#             pdf_parse = paper_id_to_pdf_parse[paper_id]
#
#             # (2) pull out fields we need from the pdf parse, including bibliography & text
#             bib_entries = pdf_parse['bib_entries']
#             paragraphs = pdf_parse['abstract']
#
#             # (3) loop over paragraphs, grabbing citation contexts
#             for paragraph in paragraphs:
