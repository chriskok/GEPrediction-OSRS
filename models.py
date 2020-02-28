from preprocessing import prepare_data, regression_f_test, recursive_feature_elim

item_to_predict = 'Rune_scimitar'
items_selected = ['Rune_axe', 'Rune_2h_sword', 'Rune_scimitar', 'Rune_chainbody', 'Rune_full_helm', 'Rune_kiteshield']

preprocessed_df = prepare_data(item_to_predict, items_selected)
print(preprocessed_df.head())
print(preprocessed_df.shape)

ftest_df = regression_f_test(preprocessed_df, item_to_predict)
print(ftest_df.head())
print(ftest_df.shape)

rfe_df = recursive_feature_elim(preprocessed_df, item_to_predict)
print(rfe_df.head())
print(rfe_df.shape)