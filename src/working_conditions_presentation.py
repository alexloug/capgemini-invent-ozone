"""English presentation charts for notebook 03; no model selection or source edits."""
from hashlib import sha256
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display
from .data import ROOT, attrition_table, export_table
from .gender_presentation import wilson

OUT = ROOT/'outputs/03_working_conditions_presentation'
BLUE, CYAN, GRAY = '#0070AD', '#12ABDB', '#7B8E9B'
CORE = {2,3,4,5,6,10,11,20}
SCORES = ['EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance','RelationshipSatisfaction']
SCORE_NAMES = dict(zip(SCORES,['Environment satisfaction','Job satisfaction','Work–life balance','Relationship satisfaction']))
TRAVEL = ['Non-Travel','Travel_Rarely','Travel_Frequently']
TRAVEL_NAMES = {'Non-Travel':'No travel','Travel_Rarely':'Rare travel','Travel_Frequently':'Frequent travel'}
TITLES = {
 1:'The dataset records 248 departures among 1,518 employees',
 2:'Overtime is associated with a higher observed departure rate',
 3:'Departure rates increase across recorded travel-frequency categories',
 4:'Recorded commuting distance is associated with departure rates',
 5:'Departure rates differ across environment-satisfaction codes',
 6:'Departure rates differ across job-satisfaction codes',
 7:'Work–life balance codes need careful, sample-aware interpretation',
 8:'Relationship-satisfaction codes show uneven sample sizes',
 9:'Overtime is associated with more departures in both gender groups',
 10:'Overtime and job-level code 1 identify a group for targeted diagnosis',
 11:'Overtime and travel should be reviewed together',
 12:'Compare distance-related departure patterns with and without overtime',
 13:'Overtime exposure differs across job roles',
 14:'Travel exposure differs across job roles',
 15:'Compare overtime within recorded environment-satisfaction codes',
 16:'Compare overtime within recorded work–life balance codes',
 17:'Distance distributions overlap between leavers and stayers',
 18:'Compare overtime-associated departures within departments',
 19:'Sparse score codes limit detailed subgroup interpretation',
 20:'Compare overtime groups on a common role-and-level composition',
}


def rates(frame, columns):
    records=[]
    grouping=columns[0] if len(columns)==1 else columns
    for key,d in frame.groupby(grouping,observed=True,dropna=False):
        key=(key,) if len(columns)==1 else key
        n,departures=len(d),int(d.depart.sum());lo,hi=wilson(departures,n)
        rate=departures/n
        # Remove floating-point overshoot at 0/1; keep error-bar lengths nonnegative.
        lo=max(0.,min(rate,lo));hi=min(1.,max(rate,hi))
        records.append(dict(zip(columns,key),n=n,departures=departures,rate=rate,low=lo,high=hi))
    return pd.DataFrame(records)


def prepare(df):
    for name in ['figures','tables']:(OUT/name).mkdir(parents=True,exist_ok=True)
    frame=df.copy(deep=True)
    frame['distance_groupe']=pd.cut(frame.DistanceFromHome,[-np.inf,5,10,20,np.inf],labels=['≤ 5',']5, 10]',']10, 20]','> 20'])
    for variable in ['OverTime','BusinessTravel',*SCORES]:
        export_table(attrition_table(frame,variable),f'03_{variable}')
    export_table(attrition_table(frame,'distance_groupe'),'03_distance')
    export_table(frame.groupby('Attrition').DistanceFromHome.agg(['count','mean','median']),'03_distance_resume',index=True)
    export_table(attrition_table(frame,['Gender','JobLevel','OverTime']),'03_overtime_strates')
    frame['DistanceBand']=pd.cut(frame.DistanceFromHome,[-np.inf,5,10,20,np.inf],labels=['≤5','>5–10','>10–20','>20'])
    frame['LevelGroup']=np.where(frame.JobLevel.eq(1),'Level code 1','Other level codes')
    # Same weights for both overtime categories, on common role × level cells only.
    counts=frame.groupby(['JobRole','JobLevel','OverTime']).size().unstack('OverTime').reindex(columns=['No','Yes'])
    common=counts.dropna().loc[lambda d:d.min(axis=1).ge(5)]
    supported=pd.MultiIndex.from_frame(frame[['JobRole','JobLevel']]).isin(common.index)
    sub=frame.loc[supported]
    weights=common.sum(axis=1)/common.sum().sum()
    rng=np.random.default_rng(42)
    means={group:0. for group in ['No','Yes']};draws={group:np.zeros(500) for group in ['No','Yes']}
    for (role,level),weight in weights.items():
        for group in ['No','Yes']:
            cell=sub[sub.JobRole.eq(role)&sub.JobLevel.eq(level)&sub.OverTime.eq(group)]
            n=len(cell);p=cell.depart.mean()
            means[group]+=weight*p
            draws[group]+=weight*rng.binomial(n,p,500)/n
    standard=[]
    for group in ['No','Yes']:
        standard.append({'overtime':group,'standardized_rate':means[group],
                         'bootstrap_low':np.quantile(draws[group],.025),'bootstrap_high':np.quantile(draws[group],.975),
                         'covered_n':len(sub),'strata':len(common),'minimum_per_overtime_group':5})
    standard=pd.DataFrame(standard)
    standard.to_csv(OUT/'tables/overtime_standardized_rates.csv',index=False)
    difference=draws['Yes']-draws['No']
    pd.DataFrame([{'standardized_rate_difference':means['Yes']-means['No'],
                   'bootstrap_low':np.quantile(difference,.025),'bootstrap_high':np.quantile(difference,.975),
                   'covered_n':len(sub),'strata':len(common)}]).to_csv(OUT/'tables/overtime_standardized_difference.csv',index=False)
    return {'df':frame,'common':sub,'standard':standard,'charts':[]}


def one_way(ax,data,column,categories,labels):
    d=data.set_index(column).reindex(categories)
    x=np.arange(len(d));ax.bar(x,100*d.rate,color=BLUE,width=.6,alpha=.85)
    ax.errorbar(x,100*d.rate,yerr=[100*(d.rate-d.low),100*(d.high-d.rate)],fmt='none',ecolor='#243746',capsize=6)
    for i,r in enumerate(d.itertuples()):ax.text(i,100*r.high+1.7,f'{100*r.rate:.1f}%\n{r.departures}/{r.n}',ha='center',fontsize=13)
    maximum=min(115,max(25,100*d.high.max()+14))
    ax.set(xticks=x,xticklabels=labels,ylabel='Observed departure rate (%)',ylim=(0,maximum))


def two_way(ax,data,column,categories,labels,group='OverTime',groups=('No','Yes'),group_names=('No overtime','Overtime')):
    for j,g in enumerate(groups):
        d=data[data[group].eq(g)].set_index(column).reindex(categories)
        for i,category in enumerate(categories):
            row=d.loc[category]
            if pd.isna(row.n):continue
            pos=i+(-.13 if j==0 else .13)
            color=(BLUE if j==0 else CYAN) if group=='Gender' else (GRAY if j==0 else BLUE)
            ax.errorbar(pos,100*row.rate,yerr=[[100*(row.rate-row.low)],[100*(row.high-row.rate)]],fmt='o',capsize=5,
                        color=color,label=group_names[j] if i==0 else None)
            ax.annotate(f'{int(row.departures)}/{int(row.n)}',(pos,100*row.high),xytext=(0,8),textcoords='offset points',ha='center',fontsize=12)
    ax.set(xticks=range(len(categories)),xticklabels=labels,ylabel='Observed departure rate (%)',ylim=(0,min(115,max(45,100*data.high.max()+15))))
    ax.legend()


def heatmap(ax,data,column,order=None):
    data=data.copy()
    data['support_ok']=data.n.ge(30)&data.departures.ge(5)&(data.n-data.departures).ge(5)
    matrix=data.pivot(index=column,columns='OverTime',values='rate').reindex(columns=['No','Yes'])
    support=data.pivot(index=column,columns='OverTime',values='support_ok').reindex_like(matrix).eq(True)
    if order is not None:matrix=matrix.reindex(order);support=support.reindex(order)
    matrix=matrix.where(support)*100
    annotations=matrix.copy().astype(object)
    for row in matrix.index:
        for col in matrix.columns:
            if pd.notna(matrix.loc[row,col]):
                cell=data[data[column].eq(row)&data.OverTime.eq(col)].iloc[0]
                annotations.loc[row,col]=f'{matrix.loc[row,col]:.1f}%\n{int(cell.departures)}/{int(cell.n)}'
            else:annotations.loc[row,col]=''
    sns.heatmap(matrix,mask=matrix.isna(),annot=annotations,fmt='',cmap='Blues',vmin=0,vmax=60,linewidths=1,
                ax=ax,annot_kws={'fontsize':15},cbar_kws={'label':'Observed departure rate (%)'})
    ax.set_xticklabels(['No overtime','Overtime'],rotation=0)
    ax.set(xlabel='',ylabel='');ax.tick_params(axis='y',rotation=0)
    return data


def chart(state,number):
    frame=state['df'];sns.set_theme(style='whitegrid',font_scale=1.25)
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(16,9));fig.subplots_adjust(left=.13,right=.92,top=.79,bottom=.22)
    fig.text(.065,.94,TITLES[number],fontsize=22,weight='bold',ha='left')
    subtitle='Observed workforce: 1,518 employees | 248 departures (16.3%) | Source values retained'
    note='Rates are departures / employees, not annual turnover. Associations do not establish causality.'
    source='Source: data/employee.csv.xlsx; src/working_conditions_presentation.py | No absence, date or actual-hours data.'
    table=None
    if number==1:
        table=pd.DataFrame({'status':['Stayed','Left'],'n':[int(frame.depart.eq(0).sum()),int(frame.depart.sum())]})
        ax.bar(table.status,table.n,color=[GRAY,BLUE],width=.5)
        for i,r in enumerate(table.itertuples()):ax.text(i,r.n+25,f'{r.n:,} | {r.n/len(frame):.1%}',ha='center',fontsize=17)
        ax.set(ylabel='Number of employees',ylim=(0,1500))
        note='Attrition Yes = departure. Observation period is unknown; absence and voluntary/involuntary departure status are not measured.'
    elif number in [2,3,4,5,6,7,8]:
        variable={2:'OverTime',3:'BusinessTravel',4:'DistanceBand',5:'EnvironmentSatisfaction',6:'JobSatisfaction',7:'WorkLifeBalance',8:'RelationshipSatisfaction'}[number]
        categories= ['No','Yes'] if number==2 else TRAVEL if number==3 else ['≤5','>5–10','>10–20','>20'] if number==4 else sorted(frame[variable].unique())
        labels=['No overtime','Overtime'] if number==2 else [TRAVEL_NAMES[c] for c in categories] if number==3 else categories
        table=rates(frame,[variable]);one_way(ax,table,variable,categories,labels)
        if number==4:ax.set_xlabel('Recorded home-to-work distance (source units; unit unknown)')
        if number>=5:
            ax.set_xlabel(f'{SCORE_NAMES[variable]}: observed score code')
            note='95% Wilson intervals; labels: departures / employees. Codes are not validated as low/high satisfaction; rare cells are uncertain.'
        else:note='95% Wilson intervals; labels: departures / employees. Unadjusted associations; no annualization or causal interpretation.'
    elif number==9:
        table=rates(frame,['OverTime','Gender'])
        two_way(ax,table,'OverTime',['No','Yes'],['No overtime','Overtime'],'Gender',('Female','Male'),('Women','Men'))
        note='95% Wilson intervals; labels: departures / employees. Gender stratification does not adjust for role, level or other differences.'
    elif number in [10,12,18]:
        variable={10:'LevelGroup',12:'DistanceBand',18:'Department'}[number]
        categories=['Level code 1','Other level codes'] if number==10 else ['≤5','>5–10','>10–20','>20'] if number==12 else sorted(frame.Department.unique())
        table=rates(frame,[variable,'OverTime']);two_way(ax,table,variable,categories,categories)
        if number==12:ax.set_xlabel('Recorded distance band (source units; not confirmed as kilometres)')
        note='95% Wilson intervals; labels: departures / employees. Descriptive strata, not causal interactions or independent replication.'
        if number==10:note+=' Level code 1 does not automatically mean junior.'
    elif number in [11,15,16]:
        variable={11:'BusinessTravel',15:'EnvironmentSatisfaction',16:'WorkLifeBalance'}[number]
        table=rates(frame,[variable,'OverTime']);table=heatmap(ax,table,variable,TRAVEL if number==11 else sorted(frame[variable].unique()))
        if number==11:ax.set_yticklabels([TRAVEL_NAMES[c] for c in TRAVEL],rotation=0)
        else:ax.set_ylabel(SCORE_NAMES[variable]+': observed score code')
        note='Blank cells: n < 30, departures < 5 or non-departures < 5. Labels: departures / employees; no causal interaction inferred.'
    elif number==13:
        table=rates(frame,['JobRole'])
        exposure=frame.groupby('JobRole').OverTime.agg(n='size',overtime_n=lambda s:s.eq('Yes').sum()).reset_index()
        exposure['overtime_share']=exposure.overtime_n/exposure.n;exposure=exposure.sort_values('overtime_share')
        lows,highs=zip(*[wilson(int(r.overtime_n),int(r.n)) for r in exposure.itertuples()])
        pos=np.arange(len(exposure));ax.errorbar(exposure.overtime_share*100,pos,xerr=[100*(exposure.overtime_share-np.array(lows)),100*(np.array(highs)-exposure.overtime_share)],fmt='o',color=BLUE,capsize=5)
        for i,r in enumerate(exposure.itertuples()):ax.text(1.02,i,f'{r.overtime_n}/{r.n}',transform=ax.get_yaxis_transform(),va='center')
        ax.set(yticks=pos,yticklabels=exposure.JobRole,xlabel='Employees reporting overtime (%)',xlim=(0,65));fig.subplots_adjust(left=.27,right=.83)
        table=exposure;note='Exposure, not departure risk. 95% Wilson intervals; labels: employees reporting overtime / employees in the role.'
    elif number==14:
        counts=pd.crosstab(frame.JobRole,frame.BusinessTravel).reindex(columns=TRAVEL,fill_value=0)
        shares=counts.div(counts.sum(axis=1),axis=0)*100
        shares.rename(columns=TRAVEL_NAMES).plot.barh(stacked=True,ax=ax,color=[GRAY,CYAN,BLUE])
        ax.set(xlabel='Share of employees within each role (%)',ylabel='',xlim=(0,100));ax.invert_yaxis()
        ax.legend(title='Recorded travel frequency',loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,fontsize=12,title_fontsize=12)
        fig.subplots_adjust(left=.27)
        table=counts.reset_index();note='Each role sums to 100%. Categories do not measure travel days; no conclusion about a travel-policy effect is possible.'
    elif number==17:
        d=frame.assign(Status=np.where(frame.depart.eq(1),'Left','Stayed'))
        sns.boxplot(data=d,x='Status',y='DistanceFromHome',order=['Stayed','Left'],hue='Status',palette={'Stayed':GRAY,'Left':BLUE},legend=False,ax=ax,width=.45)
        ax.set(xlabel='',ylabel='Recorded home-to-work distance (source units; unit unknown)')
        table=d.groupby('Status').DistanceFromHome.agg(n='size',mean='mean',median='median').reset_index()
        note='Median, quartiles and 1.5-IQR whiskers. Distance is not commuting time; overlapping distributions do not establish individual prediction accuracy.'
    elif number==19:
        counts=pd.DataFrame({SCORE_NAMES[c]:frame[c].value_counts().sort_index() for c in SCORES}).T
        sns.heatmap(counts,mask=counts.isna(),annot=True,fmt='.0f',cmap='Blues',linewidths=1,ax=ax,cbar_kws={'label':'Number of employees'})
        ax.set(xlabel='Observed score code',ylabel='');ax.tick_params(axis='y',rotation=0);fig.subplots_adjust(left=.26)
        table=counts.reset_index(names='score')
        note='Numbers are cell sample sizes, not departure counts. No code is labelled invalid without a confirmed HR dictionary.'
    elif number==20:
        standard=state['standard'].set_index('overtime').loc[['No','Yes']]
        raw=rates(state['common'],['OverTime']).set_index('OverTime').loc[['No','Yes']]
        x=np.arange(2)
        for offset,values,lows,highs,label,color in [(-.16,raw.rate,raw.low,raw.high,'Raw, common strata',GRAY),(.16,standard.standardized_rate,standard.bootstrap_low,standard.bootstrap_high,'Standardized, same strata',BLUE)]:
            ax.bar(x+offset,values*100,width=.28,label=label,color=color)
            ax.errorbar(x+offset,values*100,yerr=[100*(values-lows),100*(highs-values)],fmt='none',ecolor='#243746',capsize=5)
            for i,value in enumerate(values):ax.text(i+offset,highs.iloc[i]*100+1,f'{value:.1%}',ha='center',fontsize=14)
        ax.set(xticks=x,xticklabels=['No overtime','Overtime'],ylabel='Departure rate on common role–level strata (%)',ylim=(0,max(45,standard.bootstrap_high.max()*100+10)));ax.legend()
        subtitle=f'At least 5 employees in each overtime category per role × level cell | Coverage: {len(state["common"]):,}/1,518'
        note='Same role–level weights; 500 within-cell bootstraps for standardized rates, Wilson intervals for raw rates. Descriptive, not causal.'
        table=standard.reset_index().merge(raw.reset_index(),left_on='overtime',right_on='OverTime',suffixes=('_standard','_raw'))
    else:raise ValueError(number)
    fig.text(.065,.865,subtitle,fontsize=14,ha='left')
    fig.text(.065,.095,note,fontsize=10.3,color='#536572',ha='left')
    fig.text(.065,.052,source,fontsize=10.3,color='#536572',ha='left')
    stem=f'03_{number:02d}_'+('core' if number in CORE else 'appendix')
    for extension in ['png','svg']:fig.savefig(OUT/'figures'/f'{stem}.{extension}',dpi=300,facecolor='white')
    plt.close(fig)
    table.to_csv(OUT/'tables'/f'chart_{number:02d}_source.csv',index=False)
    state['charts'].append({'number':number,'title':TITLES[number],'priority':'Core presentation' if number in CORE else 'Appendix',
                             'png':f'figures/{stem}.png','svg':f'figures/{stem}.svg','source_table':f'tables/chart_{number:02d}_source.csv'})
    display(Image(filename=str(OUT/'figures'/f'{stem}.png')))


def finish(state,protected):
    index=pd.DataFrame(state['charts']).sort_values('number');assert index.number.nunique()==20
    index.to_csv(OUT/'chart_index.csv',index=False)
    lines=['# Working conditions — English presentation charts','',
           '20 charts, including 8 priority charts. PNG: 16:9 at 300 DPI; SVG: editable. All use the full original dataset or explicitly labelled common strata.',
           '', '| # | Priority | Chart | Files |','|---|---|---|---|']
    for row in index.itertuples():lines.append(f'| {row.number} | {row.priority} | {row.title} | [PNG]({row.png}) · [SVG]({row.svg}) · [Data]({row.source_table}) |')
    lines+=['','Rates are observed departure proportions, not annual turnover. No absence data are available. '
            'Overtime is a Yes/No category, not an hours measure. Travel categories do not measure days. '
            'Distance units and score meanings are unconfirmed. Sparse heatmap cells are masked with fixed n/event support rules. '
            'No causal interaction or discrimination finding is inferred.','',
            'Chart 20 uses shared role × level weights on cells with at least 5 employees in each overtime category. '
            'Its 500 conditional descriptive bootstraps preserve cell sizes and weights; this is not an intervention-effect estimate.','',
            'Run notebooks/03_working_conditions.ipynb to reproduce. Historical notebook-03 CSV tables are retained.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    for file,digest in protected.items():assert sha256((ROOT/file).read_bytes()).hexdigest()==digest,file
    metadata={'source':'data/employee.csv.xlsx','n':len(state['df']),'departures':int(state['df'].depart.sum()),
              'protected_sha256':protected,'module_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
              'standardization_seed':42,'standardization_bootstraps':500,'minimum_per_overtime_cell':5,
              'heatmap_rule':'n>=30; departures>=5; nondepartures>=5','new_models':False}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return index
