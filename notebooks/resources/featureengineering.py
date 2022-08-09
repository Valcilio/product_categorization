import pandas as pd

class FeatureEngineering():

    def __init__(self, df: pd.DataFrame, **kwargs):
        '''This class is specific for this problem'''

        self.df = df.copy()

    def uf(self, **kwargs):
        '''Feature who says who are the UF'''
        
        df1 = self.df
        df1['uf'] = df1['uf_cidade'].apply(lambda x: x[0:2])

        return df1['uf']

    def decade(self, **kwargs):
        '''Feature who get together a bunch of years'''

        df1 = self.df
        df1['decade'] = df1['ano_modelo'].astype(int).apply(lambda x: 
                                                            '2000 <' if x < 2000 else
                                                            '2000 > | 2005 <' if (x > 2000) & (x < 2005) else
                                                            '2005 > | 2010 <' if (x > 2005) & (x < 2010) else
                                                            '2010 > | 2015 <' if (x > 2010) & (x < 2015) else
                                                            '2015 > | 2020 <' if (x > 2015) & (x < 2020) else
                                                            '2020 >')

        return df1['decade']

    def ipva_dono(self, **kwargs):
        '''Feature who get together the IPVA paid feature and with only one owner feature'''

        df1 = self.df
        df1['unico_dono'] = df1['flg_unico_dono'].apply(lambda x: 'oneowner' if x == '1' else 'moreowners')
        df1['ipva_pago'] = df1['flg_ipva_pago'].apply(lambda x: 'paid' if x == '1.0' else 'no_paid')
        df1['ipva_dono'] = df1['ipva_pago'] + '_|_' + df1['unico_dono']
        
        return df1['ipva_dono']

    def sensors(self, **kwargs):
        '''Feature who indicates if exist sensors in the car'''

        df1 = self.df
        df1['schuva'] = df1['sensorchuva'].apply(lambda x: 'schuva' if x == 'S' else 'no_schuva')
        df1['srestacion'] = df1['sensorestacion'].apply(lambda x: 'restacion' if x == 'S' else 'no_rest')
        df1['sensors'] = df1['schuva'] + '_|_' + df1['srestacion']

        return df1['sensors']

    def eletr_car(self, **kwargs):
        '''Feature who indicates if exist eletricfy features in the car'''

        df1 = self.df
        df1['el_trava'] = df1['travaeletr'].apply(lambda x: 'travaeletr' if x == 'S' else 'no_travaeletr')
        df1['el_vidro'] = df1['vidroseletr'].apply(lambda x: 'vidroseletr' if x == 'S' else 'no_vidroseletr')
        df1['eletr_car'] = df1['el_trava'] + '_|_' + df1['el_vidro']

        return df1['eletr_car']

    def best_offer(self, **kwargs):
        '''Feature who indicates if the car have the best offer possible'''

        df1 = self.df
        df1['prioridade'] = df1['prioridade'].apply(lambda x: 'prio' if x == '2' else 'no_prio')
        df1['flg_aceita_troca'] = df1['flg_aceita_troca'].apply(lambda x: 'troca' if x == '1' else 'no_troca')
        df1['flg_blindado'] = df1['flg_blindado'].apply(lambda x: 'blind' if x == '1' else 'no_blind')
        df1['flg_todas_revisoes_agenda_veiculo'] = df1['flg_todas_revisoes_agenda_veiculo'].apply(lambda x: 'all_rev' if x == '1' else 'no_rev')

        df1['big_offer'] = (df1['prioridade'] + '_|_' + df1['flg_aceita_troca'] + '_|_' + df1['flg_blindado'] + '_|_' + df1['flg_todas_revisoes_agenda_veiculo'])

        first_offers = ['no_prio_|_troca_|_blind_|_all_rev', 'no_prio_|_troca_|_blind_|_no_rev',
               'prio_|_no_troca_|_blind_|_all_rev', 'prio_|_troca_|_blind_|_all_rev']

        second_offers = ['no_prio_|_no_troca_|_blind_|_all_rev', 'no_prio_|_troca_|_no_blind_|_all_rev',
                        'no_prio_|_troca_|_no_blind_|_no_rev', 'prio_|_no_troca_|_blind_|_no_rev',
                        'prio_|_no_troca_|_no_blind_|_all_rev', 'prio_|_troca_|_blind_|_no_rev',
                        'prio_|_troca_|_no_blind_|_all_rev', 'prio_|_troca_|_no_blind_|_no_rev']

        third_offers = ['no_prio_|_no_troca_|_blind_|_no_rev', 'no_prio_|_no_troca_|_no_blind_|_all_rev',
                        'no_prio_|_no_troca_|_no_blind_|_no_rev', 'prio_|_no_troca_|_no_blind_|_no_rev']

        df1['best_offer'] = df1['big_offer'].apply(lambda x: 1 if x in first_offers else
                                                    2 if x in second_offers else
                                                    3 if x in third_offers else 
                                                    3)

        return df1['best_offer']
