"""English, slide-ready gender charts for notebook 02; historical exports retained."""
from pathlib import Path
from hashlib import sha256
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display
from .data import ROOT, export_table, attrition_table

OUT = ROOT / 'outputs/02_gender_presentation'
ORDER = ['Female', 'Male']
COLORS = {'Female': '#0070AD', 'Male': '#12ABDB'}
DISPLAY = {'Female': 'Women', 'Male': 'Men'}
CORE = {2, 5, 8, 10, 11, 12}
TITLES = {
 1: 'Women account for {share:.1%} of the observed workforce',
 2: 'Observed departure rates differ by gender',
 3: 'Monthly income distributions overlap substantially',
 4: 'Compare both mean and median monthly income',
 5: 'Gender groups have different job-level compositions',
 6: 'Job-role composition differs between women and men',
 7: 'Women’s representation varies across job levels',
 8: 'Women’s representation varies across job roles',
 9: 'Compare monthly income within each observed job level',
 10: 'The raw pay gap changes after role-and-level standardization',
 11: 'Comparable-stratum pay gaps vary across roles and levels',
 12: 'Overtime is associated with higher departure rates in both gender groups',
 13: 'Departure rates by level require attention to sample size',
 14: 'Recorded salary increases describe outcomes, not access to raises',
 15: 'Career timing indicators do not measure promotion opportunities',
 16: 'Training participation can be compared across gender groups',
 17: 'Standardized pay estimates depend on the coverage rule',
 18: 'Gender representation should be examined within role-and-level cells',
}


def wilson(departures, n):
    p = departures/n
    z = 1.959963984540054
    mid = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return mid-radius, mid+radius


def prepare(df):
    """Keep all historical notebook-02 tables and add flat English chart sources."""
    for folder in ['figures', 'tables']:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    export_table(attrition_table(df, 'Gender'), '02_depart_genre')
    variables = ['MonthlyIncome', 'PercentSalaryHike', 'JobLevel', 'YearsAtCompany',
                 'YearsInCurrentRole', 'YearsSinceLastPromotion']
    export_table(df.groupby('Gender')[variables].agg(['count','mean','median']), '02_parcours_genre', index=True)
    for col in ['JobLevel', 'JobRole']:
        counts = pd.crosstab(df.Gender, df[col])
        export_table(counts, f'02_effectifs_{col}', index=True)
        export_table(counts.div(counts.sum(axis=1), axis=0), f'02_parts_{col}', index=True)
    strata = df.groupby(['JobRole','JobLevel','Gender'], observed=True).agg(
        effectif=('depart','size'), departs=('depart','sum'), proportion_depart=('depart','mean'),
        revenu_moyen=('MonthlyIncome','mean'), revenu_median=('MonthlyIncome','median'),
        delai_promotion_moyen=('YearsSinceLastPromotion','mean')).reset_index()
    export_table(strata,'02_strates_metier_niveau')
    incomes = strata.pivot(index=['JobRole','JobLevel'], columns='Gender', values='revenu_moyen')
    counts = strata.pivot(index=['JobRole','JobLevel'], columns='Gender', values='effectif')
    comparable = incomes.dropna(subset=ORDER).copy()
    comparable['ecart_femmes_hommes_pct'] = 100*(comparable.Female/comparable.Male-1)
    comparable = comparable.join(counts.add_prefix('n_'))
    export_table(comparable,'02_ecarts_comparables',index=True)
    common = strata.set_index(['JobRole','JobLevel']).loc[comparable.index].reset_index()
    weights = common.groupby(['JobRole','JobLevel']).effectif.sum()
    weights = (weights/weights.sum()).rename('poids')
    common = common.join(weights,on=['JobRole','JobLevel'])
    standard = common.assign(revenu_pondere=common.revenu_moyen*common.poids,
                             depart_pondere=common.proportion_depart*common.poids)
    export_table(standard.groupby('Gender')[['revenu_pondere','depart_pondere']].sum(),'02_standardisation',index=True)
    strata.to_csv(OUT/'tables/role_level_gender_strata.csv',index=False)
    summary = []
    for gender, d in df.groupby('Gender'):
        low, high = wilson(int(d.depart.sum()),len(d))
        summary.append({'gender':gender,'n':len(d),'departures':int(d.depart.sum()),'departure_rate':d.depart.mean(),
                        'wilson_low':low,'wilson_high':high,'mean_income':d.MonthlyIncome.mean(),'median_income':d.MonthlyIncome.median()})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT/'tables/gender_summary.csv',index=False)
    records = []
    for minimum in [1,5,10]:
        index = counts.dropna().loc[lambda x: x.min(axis=1).ge(minimum)].index
        sub = common.set_index(['JobRole','JobLevel']).loc[index].reset_index()
        w = sub.groupby(['JobRole','JobLevel']).effectif.sum()
        w = w/w.sum()
        means = incomes.loc[index].mul(w,axis=0).sum()
        selected = pd.MultiIndex.from_frame(df[['JobRole','JobLevel']]).isin(index)
        raw_common = df.loc[selected].groupby('Gender').MonthlyIncome.mean()
        records.append({'minimum_per_gender':minimum,'strata':len(index),'covered_n':int(sub.effectif.sum()),
                        'total_n':len(df),'women_mean_standardized':means.Female,'men_mean_standardized':means.Male,
                        'standardized_gap_percent':100*(means.Female/means.Male-1),
                        'raw_common_gap_percent':100*(raw_common.Female/raw_common.Male-1)})
    sensitivity = pd.DataFrame(records)
    sensitivity.to_csv(OUT/'tables/pay_standardization_sensitivity.csv',index=False)
    # Fixed stratum weights, resampling within gender/stratum; conditional descriptive uncertainty.
    rng = np.random.default_rng(42)
    index = counts.dropna().loc[lambda x: x.min(axis=1).ge(5)].index
    covered = pd.MultiIndex.from_frame(df[['JobRole','JobLevel']]).isin(index)
    sub = df.loc[covered]
    n = sub.groupby(['JobRole','JobLevel']).size(); w = n/n.sum()
    standardized = {g:np.zeros(500) for g in ORDER}
    for (role,level), weight in w.items():
        for gender in ORDER:
            values = sub.loc[sub.JobRole.eq(role)&sub.JobLevel.eq(level)&sub.Gender.eq(gender),'MonthlyIncome'].to_numpy()
            standardized[gender] += weight*rng.choice(values,(500,len(values)),replace=True).mean(axis=1)
    pay_rows = []
    for label, data in [('Full workforce: raw',df),('Common strata: raw',sub)]:
        draws = {}
        means = data.groupby('Gender').MonthlyIncome.mean()
        for gender in ORDER:
            values = data.loc[data.Gender.eq(gender),'MonthlyIncome'].to_numpy()
            draws[gender] = rng.choice(values,(500,len(values)),replace=True).mean(axis=1)
        gap = 100*(draws['Female']/draws['Male']-1)
        pay_rows.append({'comparison':label,'gap_percent':100*(means.Female/means.Male-1),
                         'bootstrap_low':np.quantile(gap,.025),'bootstrap_high':np.quantile(gap,.975),'n':len(data)})
    gap = 100*(standardized['Female']/standardized['Male']-1)
    point = sensitivity[sensitivity.minimum_per_gender.eq(5)].iloc[0]
    pay_rows.append({'comparison':'Common strata: standardized','gap_percent':point.standardized_gap_percent,
                     'bootstrap_low':np.quantile(gap,.025),'bootstrap_high':np.quantile(gap,.975),'n':len(sub)})
    pay = pd.DataFrame(pay_rows)
    pay.to_csv(OUT/'tables/pay_gap_comparison.csv',index=False)
    return {'df':df,'summary':summary,'strata':strata,'counts':counts,'incomes':incomes,
            'sensitivity':sensitivity,'pay':pay,'charts':[]}


def chart(state, number):
    df = state['df']; summary = state['summary'].set_index('gender').loc[ORDER]
    sns.set_theme(style='whitegrid',font_scale=1.15)
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(left=.12,right=.93,top=.79,bottom=.22)
    share = df.Gender.eq('Female').mean()
    title = TITLES[number].format(share=share)
    fig.text(.065,.94,title,fontsize=22,weight='bold',ha='left')
    subtitle = 'Observed workforce: 1,518 employees | Women: 605 | Men: 913'
    foot = 'Source: data/employee.csv.xlsx; src/gender_presentation.py | Observational data; no causal or discrimination claim.'
    extra = 'Departure proportions are not annual rates. Currency, observation period and score definitions are not documented.'
    if number == 1:
        ax.bar(['Women','Men'],summary.n,color=list(COLORS.values()),width=.5)
        for i,r in enumerate(summary.itertuples()):ax.text(i,r.n+20,f'{r.n:,}\n{r.n/len(df):.1%}',ha='center',fontsize=17)
        ax.set(ylabel='Number of employees',ylim=(0,1100))
    elif number == 2:
        ax.bar(['Women','Men'],100*summary.departure_rate,color=list(COLORS.values()),width=.5)
        ax.errorbar([0,1],100*summary.departure_rate,
                    yerr=[100*(summary.departure_rate-summary.wilson_low),100*(summary.wilson_high-summary.departure_rate)],
                    fmt='none',ecolor='#243746',capsize=8)
        for i,r in enumerate(summary.itertuples()):ax.text(i,100*r.wilson_high+1,f'{r.departure_rate:.1%}\n{r.departures}/{r.n} departures',ha='center',fontsize=16)
        ax.set(ylabel='Observed departure rate (%)',ylim=(0,25))
        extra='Bars: departures / employees; whiskers: 95% Wilson intervals. No adjustment for job, level or other characteristics.'
    elif number in [3,14]:
        variable = 'MonthlyIncome' if number==3 else 'PercentSalaryHike'
        sns.boxplot(data=df,x='Gender',y=variable,order=ORDER,hue='Gender',palette=COLORS,legend=False,ax=ax,width=.45)
        ax.set_xticks([0,1],['Women (n=605)','Men (n=913)'])
        ax.set(xlabel='',ylabel='Monthly income (source units; currency unknown)' if number==3 else 'Recorded salary increase (%)')
        extra='Median, quartiles and 1.5-IQR whiskers; points outside whiskers are retained. Salary-increase period is undocumented.'
    elif number == 4:
        from matplotlib.patches import Patch
        x=np.arange(2)
        for offset,col,hatch in [(-.18,'mean_income',''),(.18,'median_income','///')]:
            ax.bar(x+offset,summary[col],width=.32,color=list(COLORS.values()),hatch=hatch,edgecolor='#243746',linewidth=.5)
            for i,value in enumerate(summary[col]):ax.text(i+offset,value+100,f'{value:,.0f}',ha='center',fontsize=14)
        ax.set(xticks=x,xticklabels=['Women','Men'],ylabel='Monthly income (source units; currency unknown)',ylim=(0,8000))
        ax.legend(handles=[Patch(facecolor='white',edgecolor='#243746',label='Mean'),
                           Patch(facecolor='white',edgecolor='#243746',hatch='///',label='Median')])
    elif number in [5,6]:
        variable='JobLevel' if number==5 else 'JobRole'
        counts=pd.crosstab(df[variable],df.Gender).reindex(columns=ORDER,fill_value=0)
        proportions=counts.div(counts.sum(axis=0),axis=1)*100
        if number==5:
            proportions.rename(columns=DISPLAY).plot.bar(ax=ax,color=list(COLORS.values()),rot=0)
            for patch,value in zip(ax.patches,proportions.T.to_numpy().ravel()):ax.annotate(f'{value:.1f}%',(patch.get_x()+patch.get_width()/2,patch.get_height()),ha='center',va='bottom',fontsize=11)
            ax.set(xlabel='Observed job-level code',ylabel='Share within each gender group (%)',ylim=(0,48))
        else:
            proportions.rename(columns=DISPLAY).plot.barh(ax=ax,color=list(COLORS.values()))
            fig.subplots_adjust(left=.25)
            ax.set(xlabel='Share within each gender group (%)',ylabel='Job role');ax.invert_yaxis()
        extra='Each gender group sums to 100%. Composition is not a hiring or promotion rate. Job-level codes require HR validation.'
        counts.reset_index().to_csv(OUT/'tables'/f'{variable}_counts.csv',index=False)
    elif number in [7,8]:
        variable='JobLevel' if number==7 else 'JobRole'
        counts=pd.crosstab(df[variable],df.Gender).reindex(columns=ORDER,fill_value=0)
        rows=[]
        for category,r in counts.iterrows():
            n=int(r.sum()); low,high=wilson(int(r.Female),n)
            rows.append({'category':str(category),'n':n,'female_n':int(r.Female),'share':r.Female/n,'low':low,'high':high})
        data=pd.DataFrame(rows)
        if number==8:data=data.sort_values('share')
        pos=np.arange(len(data))
        ax.errorbar(data.share*100,pos,xerr=[(data.share-data.low)*100,(data.high-data.share)*100],fmt='o',color=COLORS['Female'],capsize=5)
        for i,r in enumerate(data.itertuples()):ax.text(1.02,i,f'{r.female_n}/{r.n} | {r.share:.1%}',transform=ax.get_yaxis_transform(),va='center',fontsize=12)
        ax.axvline(share*100,color='#777777',linestyle='--',label=f'Overall women’s share: {share:.1%}')
        ax.set(yticks=pos,yticklabels=data.category,xlim=(0,100),xlabel='Women as a share of employees in each group (%)',ylabel='Job-level code' if number==7 else '')
        fig.subplots_adjust(left=.26,right=.78);ax.legend(loc='lower right')
        extra='Whiskers: 95% Wilson intervals. Representation describes this file, not equal opportunity. Small groups remain uncertain.'
        data.to_csv(OUT/'tables'/f'{variable}_female_representation.csv',index=False)
    elif number == 9:
        sns.boxplot(data=df,x='JobLevel',y='MonthlyIncome',hue='Gender',hue_order=ORDER,palette=COLORS,ax=ax)
        counts=pd.crosstab(df.JobLevel,df.Gender)
        ax.set_xticks(range(len(counts)),[f'{level}\nW={int(r.Female)}, M={int(r.Male)}' for level,r in counts.iterrows()])
        ax.set(xlabel='Observed job-level code | W: women, M: men',ylabel='Monthly income (source units; currency unknown)')
        handles,labels=ax.get_legend_handles_labels();ax.legend(handles,[DISPLAY[x] for x in labels],title='Gender')
        extra='Distributions are conditional on level only, not job role, hours or experience. Code 0 and small groups are retained explicitly.'
    elif number == 10:
        pay=state['pay'];pos=np.arange(len(pay))
        ax.hlines(pos,pay.bootstrap_low,pay.bootstrap_high,color='#243746',linewidth=3)
        ax.plot(pay.gap_percent,pos,'o',color='#0070AD',markersize=11)
        for i,r in enumerate(pay.itertuples()):ax.text(1.02,i,f'{r.gap_percent:+.2f}% | n={r.n:,}',transform=ax.get_yaxis_transform(),va='center',fontsize=14)
        ax.set(yticks=pos,yticklabels=pay.comparison,xlabel='Mean income gap: (women / men − 1) × 100 (%)',ylim=(2.5,-.5));ax.axvline(0,color='gray',linestyle='--')
        fig.subplots_adjust(left=.30,right=.76)
        coverage=state['sensitivity'].query('minimum_per_gender == 5').iloc[0]
        subtitle=f'Common strata: same job role × level; at least 5 women and 5 men per cell | Coverage: {int(coverage.covered_n):,}/{len(df):,}'
        extra='500 descriptive bootstraps; standardization uses fixed common-stratum weights. Not adjusted for working hours or experience.'
    elif number in [11,18]:
        counts=state['counts']; enough=counts.min(axis=1).ge(5) if number==11 else counts.sum(axis=1).ge(10)
        value=100*(state['incomes'].Female/state['incomes'].Male-1) if number==11 else 100*counts.Female.fillna(0)/counts.fillna(0).sum(axis=1)
        matrix=value.where(enough).unstack('JobLevel').reindex(sorted(df.JobRole.unique()))
        annotations=matrix.copy().astype(object)
        for role in matrix.index:
            for level in matrix.columns:
                if pd.notna(matrix.loc[role,level]):
                    r=counts.loc[(role,level)].fillna(0)
                    annotations.loc[role,level]=f'{matrix.loc[role,level]:+.1f}%\nW={int(r.Female)}, M={int(r.Male)}' if number==11 else f'{matrix.loc[role,level]:.0f}%\nn={int(r.sum())}'
                else:annotations.loc[role,level]=''
        sns.heatmap(matrix,mask=matrix.isna(),annot=annotations,fmt='',cmap='RdBu_r' if number==11 else 'Blues',
                    center=0 if number==11 else None,vmin=None if number==11 else 0,vmax=None if number==11 else 100,
                    annot_kws={'fontsize':11},linewidths=1,ax=ax,cbar_kws={'label':'Income gap: women vs men (%)' if number==11 else 'Women (%)'})
        ax.set(xlabel='Observed job-level code',ylabel='');ax.tick_params(axis='y',rotation=0)
        fig.subplots_adjust(left=.27,right=.93,bottom=.22)
        extra=('Blank cells: missing group or fewer than 5 employees of either gender. Descriptive gaps, not discrimination estimates; no cell-level CIs.' if number==11 else
               'Blank cells: fewer than 10 employees or no observed role–level cell. Representation is not a measure of promotion opportunity.')
        matrix.to_csv(OUT/'tables'/f'chart_{number:02d}_heatmap.csv')
    elif number in [12,13]:
        variable='OverTime' if number==12 else 'JobLevel'
        categories=['No','Yes'] if number==12 else sorted(df.JobLevel.unique())
        rows=[]
        for j,gender in enumerate(ORDER):
            for i,category in enumerate(categories):
                d=df[df.Gender.eq(gender)&df[variable].eq(category)];n=len(d)
                if not n:continue
                dep=int(d.depart.sum());rate=dep/n;low,high=wilson(dep,n);pos=i+(-.12 if j==0 else .12)
                ax.errorbar(pos,100*rate,yerr=[[100*(rate-low)],[100*(high-rate)]],fmt='o',capsize=4,
                            color=COLORS[gender],label=DISPLAY[gender] if i==0 else None)
                ax.annotate(f'{dep}/{n}',(pos,high*100),xytext=(0,8),textcoords='offset points',ha='center',fontsize=11)
                rows.append({'gender':gender,'category':category,'n':n,'departures':dep,'rate':rate,'low':low,'high':high})
        ax.set(xticks=range(len(categories)),xticklabels=['No overtime','Overtime'] if number==12 else categories,
               xlabel='' if number==12 else 'Observed job-level code',ylabel='Observed departure rate (%)',ylim=(0,100 if number==13 else 50));ax.legend()
        extra='Whiskers: 95% Wilson intervals; labels: departures / employees. Sparse cells are uncertain; comparisons are unadjusted.'
        pd.DataFrame(rows).to_csv(OUT/'tables'/f'chart_{number:02d}_departure_rates.csv',index=False)
    elif number == 15:
        ax.remove();axes=fig.subplots(1,3)
        for axis,variable,label in zip(axes,['YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion'],['Years at company','Years in current role','Years since last promotion']):
            sns.boxplot(data=df,x='Gender',y=variable,hue='Gender',order=ORDER,palette=COLORS,legend=False,ax=axis,width=.45)
            axis.set(xticks=[0,1],xticklabels=['Women','Men'],xlabel='',ylabel='Years (recorded)',title=label)
        extra='Time since last promotion is not a promotion rate. Tenure inconsistencies are retained; no access-to-promotion conclusion is possible.'
    elif number == 16:
        counts=pd.crosstab(df.TrainingTimesLastYear,df.Gender).reindex(columns=ORDER)
        counts.div(counts.sum(axis=0),axis=1).mul(100).rename(columns=DISPLAY).plot.bar(ax=ax,color=list(COLORS.values()),rot=0)
        ax.set(xlabel='Recorded training count last year',ylabel='Share within each gender group (%)')
        extra='Counts do not measure training quality, eligibility or hours. TrainingTimesLastYear is used as recorded.'
        counts.reset_index().to_csv(OUT/'tables/training_counts.csv',index=False)
    elif number == 17:
        data=state['sensitivity'];pos=np.arange(len(data))
        ax.plot(pos,data.standardized_gap_percent,'o-',color='#0070AD',markersize=10)
        for i,r in enumerate(data.itertuples()):ax.annotate(f'{r.standardized_gap_percent:+.2f}%\nCoverage: {r.covered_n:,}/{r.total_n:,}\n{r.strata} strata',
                    (i,r.standardized_gap_percent),xytext=(0,20),textcoords='offset points',ha='center',fontsize=13)
        ax.set(xticks=pos,xticklabels=[f'At least {n} of each gender' for n in data.minimum_per_gender],
               xlabel='Minimum sample size per gender in a role × level cell',ylabel='Standardized mean income gap: women vs men (%)',ylim=(min(-3,data.standardized_gap_percent.min()-1),1))
        ax.axhline(0,color='gray',linestyle='--')
        extra='Sensitivity analysis, not a confidence interval. Changing the threshold changes both coverage and population weights.'
    else:raise ValueError(number)
    fig.text(.065,.865,subtitle,fontsize=14,ha='left')
    fig.text(.065,.095,extra,fontsize=10.5,ha='left',color='#536572')
    fig.text(.065,.052,foot,fontsize=10.5,ha='left',color='#536572')
    stem=f'02_{number:02d}_'+('core' if number in CORE else 'appendix')
    for ext in ['png','svg']:fig.savefig(OUT/'figures'/f'{stem}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    state['charts'].append({'number':number,'title':title,'priority':'Core presentation' if number in CORE else 'Appendix',
                             'png':f'figures/{stem}.png','svg':f'figures/{stem}.svg','population':'Full dataset, 1518 employees; filtered subsets explicitly labelled'})
    display(Image(filename=str(OUT/'figures'/f'{stem}.png')))


def finish(state, protected):
    index=pd.DataFrame(state['charts']).sort_values('number')
    assert index.number.nunique()==18
    index.to_csv(OUT/'chart_index.csv',index=False)
    lines=['# Gender equality — English presentation charts','',
           '18 charts, 16:9, PNG at 300 DPI and editable SVG. Start with the six **Core presentation** charts. '
           'All charts use the full original dataset; do not confuse these figures with training-only results in reports/.','',
           '| # | Priority | Chart | Files |','|---|---|---|---|']
    for row in index.itertuples():lines.append(f'| {row.number} | {row.priority} | {row.title} | [PNG]({row.png}) · [SVG]({row.svg}) |')
    lines+=['','Income units, observation period and score definitions are undocumented. '
            'Observed departures are not annual turnover rates. Pay gaps are descriptive, not discrimination findings; '
            'standardization controls only recorded role and level. Sparse cells are masked or flagged. '
            'No promotion rate can be inferred from time since last promotion.','',
            'Reproduce by running notebooks/02_gender_equality.ipynb. Historical notebook-02 CSV exports are retained. '
            'The new descriptive bootstrap uses 500 repetitions and fixed common-stratum weights.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    for path,digest in protected.items():assert sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    metadata={'source':'data/employee.csv.xlsx','source_n':len(state['df']),'departures':int(state['df'].depart.sum()),
              'women_n':605,'men_n':913,'common_strata_minimum_per_gender':5,'descriptive_bootstraps':500,
              'protected_sha256':protected,'module_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
              'comparability_note':'Full-data descriptives; reports/ contains separate training-only standardization.'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return index
