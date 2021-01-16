
import pandas as pd 
import streamlit as st
#import sidetable as stb 
#from pandas_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
import tsc_helper as th
import altair as alt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer

st.set_page_config(layout='wide')
data, key = th.read_data(uploaded_files=True)
#data, key = th.read_data(uploaded_files=True)
navigation = st.sidebar.selectbox(label='Navigation',options=['Introduction','Sample Information', 'Group Rankings', 'EFA', 'Clustering','Columns and Key', 'Longitudinal', 'Correlations', 'Key'], index=0)


#data.loc[data.ResearcherName == 'CCC Researcher', :]

if isinstance(data, str):
    st.write('Please upload data files in the file selector above')

else:

    items = list(data.loc[:, 'TheoreticalOrientation.1':'Interventions.3120'])
    researcherList = list(data['ResearcherName'].unique())
    researcherList.sort()

    if navigation == 'Introduction':
        st.title('Introduction')

    if navigation == 'Group Rankings':
        
        #Adding controls
        showAllResearchers = st.sidebar.checkbox(
        'Show data for all researchers (may take some time to load)',
        value= True )

        selectedResearcher = st.sidebar.multiselect('Researchers', researcherList)


        selectedSection = st.sidebar.selectbox(
            label= 'TSC Section',
            options = [ 'Theoretical Orientation', 'Interventions', 'Spiritual Interventions', 'Counseling Topics', 'Intentions']
        )
        groupingList = ['ResearcherName', 'LocationName','LocationType', 'Modality',
        'SessionCount', 'EthnicityCategorized', 'ReligiousAffiliation', 'IsReligionImportant', 'TherapistJoined', 'ClientName']
        groupingVariable = st.sidebar.selectbox(label='Grouping Variable',
        options= groupingList
        )



        #Filtering data
        st.write(th.vc_plus(data,'TheoreticalOrientation.2'))

        dataFiltered = th.filter_researcher(data =data, researcher= selectedResearcher, showAll = showAllResearchers, sections = selectedSection)
        dataFiltered = th.filter_section(dataFiltered, selectedSection, grouping_variable=groupingVariable)
        
        st.title(f'Rankings by {groupingVariable}')
        #Adding selector for table
        toggleTable = st.checkbox('Show rank tables')

        key['VariableId'] = key['VariableId'].str.replace(' ', '')
        meansFiltered = th.tsc_means(dataFiltered, key)

        itemsFiltered = [i for i in dataFiltered.columns]

        

        #st.write(meansFiltered)
        
        #maxRanking = st.number_input(label='Max Ranking', value=3)
        groupPivot = th.grouped_pivot_table(dataFiltered, key, groupingVariable)
        groupPivotNoPercent = th.grouped_pivot_table(dataFiltered, key, groupingVariable, include_percents=False)
        
        #Displays top options based on them being in the top 3 counts
        th.top_values(groupPivotNoPercent, tables=toggleTable)

        
        groupPivotExpander = st.beta_expander(label='Show full table')
        with groupPivotExpander:
            st.table(groupPivot)

    elif navigation == 'Sample Information':
        st.title('Sample Information')

        sampleCategory = st.sidebar.selectbox('Select Sample Categories to view', options= [
            'All'
            ,'Site Characteristics'
            ,'Client Characteristics'
            ,'Treatment Characteristics'
            ,'Missing Data'
        ])

        # groupingVariable = st.sidebar.selectbox(label='Grouping Variable',
        # options=['ResearcherName', 'LocationName','LocationType', 'Modality',
        # 'SessionCount', 'EthnicityCategorized', 'ReligiousAffiliation', 'IsReligionImportant', 'TherapistJoined']
        # )
        metric = st.sidebar.selectbox('Select metric for descriptives', options=['Responses', 'Clients'])
        if metric == 'Clients':
            data = data.drop_duplicates(['ResearcherName', 'ClientName'])
        
        excludedResearchers = st.sidebar.multiselect('Exclude Researchers', options = researcherList)
        if excludedResearchers:
            data = data.loc[~data['ResearcherName'].isin(excludedResearchers)]
        
        #st.header('Basic Stats')
        totalClients = len(data['ClientName'].unique())
        st.write(f'Total Sample Size: {str(data.shape[0])}, Total Clients: {totalClients}')
        
        #Stopped here, trying to figure out a way to isolate the unique therapists for ones that have multiple
        data['SessionMax'] = data.groupby('ClientName')['SessionCount'].transform('max')
        

        if (sampleCategory == 'All'):
            th.site_characteristics(data)
            st.markdown('----')
            th.client_characteristics(data)
            st.markdown('---')
            th.treatment_characteristics(data)
            st.markdown('---')
            th.missing_data_characteristics(data)
        elif (sampleCategory == 'Site Characteristics'):
            th.site_characteristics(data)
        elif (sampleCategory == 'Client Characteristics'):
            th.client_characteristics(data)
        elif (sampleCategory == 'Treatment Characteristics'):
            th.treatment_characteristics(data)
        elif (sampleCategory == 'Missing Data'):
            th.missing_data_characteristics(data)

        

        #Might want to add stuff about missing data, I am thinking I can categorize it to be either 1 or 0, 1 meaning there are responses for that section, 0 being no responses
        #Although I have the "sum empty section" variable which could just do that
        
        #st.write(data['IntentionsNotNa'].value_counts())
        #st.write(data.loc[data['IntentionsNotNa'] == 0, [i for i in data.columns if 'Intentions' in i]])
        notNaCols = [i for i in data.columns if 'NotNa' in i]
        
    


    elif navigation == 'Columns and Key':
        c1,c2 = st.beta_columns((1,1))
        c1.table(data.columns)
        c2.table(key[['VariableId', 'ItemText']])

    elif navigation == 'EFA':
        #Creating interactive widgets

        data = th.researcher_select(data, researcherList)
        #st.write(data.ResearcherName.value_counts())
       
        selectedSection = st.sidebar.selectbox(
            label= 'TSC Section for FA',
            options = [ 'Theoretical Orientation', 'Interventions', 'Spiritual Interventions', 'Counseling Topics', 'Intentions']
            )
        selectedSection = selectedSection.replace(' ', '')

        groupToggle = st.sidebar.checkbox('Group values by client', value=True) 
        groupingList = ['ResearcherName', 'LocationName','LocationType', 'Modality',
        'SessionCount', 'EthnicityCategorized', 'ReligiousAffiliation', 'IsReligionImportant', 'TherapistJoined', 'ClientName']
        efaGrouping = st.sidebar.selectbox('Grouping for percentage charts', groupingList)

        if data.empty:
            st.write('Please select a researcher to include, or select all researchers using the toggle on the left')
        else:
            items = [i for i in data.columns if (selectedSection in i) & ('NotNa' not in i)]
                    
            if groupToggle == True:
                itemsClient = items + ['ClientName']
                itemData = data[itemsClient]
                itemData = itemData.groupby('ClientName')[items].mean()
            else:
                itemData = data[items]

            itemData.head()
            #Doing initial model tets
            #th.efa_model_tests(itemData)
            st.write(itemData.shape)
            #Determining number of factors
            st.header('Factor Selection')
            
            runScree = st.button('Create Screeplot')
            th.scree_plot(itemData, items)
  

            #Reading output
            st.header('EFA Model')
            c1, c2, c3 = st.beta_columns(3)
            nFactors = int(c1.number_input('Number of Factors', value=0, step=1))
            onlyShowHighest = c3.checkbox('Only show loadings for highest factor', value=True)
            loadingCutoff = c2.number_input('Minimum factor loading', value=.40)
            #runEfa = st.button('Run Analysis')
            if nFactors != 0:
                efa = FactorAnalyzer(nFactors)
                efa.fit(itemData)
                loadings = th.rename_loadings(efa.loadings_, key, itemData)
                st.header('Output')
                #export1, export2 = st.beta_columns(2)
                
                
                allTables = th.separate_factor_output(loadings, onlyShowHighest, loadingCutoff)
                loadingExpander = st.beta_expander('Model Information',expanded=True)
                dataScored = th.score_factors(data, allTables, key)
                factorCols = [i for i in dataScored.columns if 'Mean' in i]
                #st.write(dataScored.groupby('ResearcherName')[factorCols].mean().reset_index())
                chartList = th.create_factor_charts(dataScored, efaGrouping, 2)
                
                with loadingExpander:
                    th.fa_display(allTables, chartList, nFactors, columns = 2)
                export_expander = st.beta_expander('Export')

                st.header('Factor Correlations')
                st.write(dataScored[factorCols].corr())
                st.write(itemData.columns)

                with export_expander:
                    exportFileName = st.text_input('Export File Name (Will export when you press enter)')
                    exportToggle = st.button('Export')
                if exportToggle:
                    th.export_loadings(allTables, exportFileName)
                
                


                
            else:
                pass
    
    if navigation == 'Clustering':
        st.header('Clustering')


    else:
        pass


    #st.write(dataFiltered.columns)
    #st.title('Pandas profiler')
    #profiler = ProfileReport(dataFiltered, minimal=True)
    #st_profile_report(profiler)