"""Descriptive environmental figures and QC/weather tables."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..data.profiler_qc import RULES
from ..data.weather_sources import source_records,VARIABLES
from ..data.qualified import substitute
from ..evaluate.references import seasonal_window_mean
from .publication import atomic_text,save_figure,tex,minus





def environmental(raw,prepared,out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    figures=out/'figures';tables=out/'tables';figures.mkdir(parents=True,exist_ok=True);tables.mkdir(exist_ok=True)
    start,end=pd.Timestamp('2021-12-13'),pd.Timestamp('2025-02-10 12:00')
    fig,axes=plt.subplots(7,1,figsize=(11.5,5),sharex=True)
    labels=['Temperature\n(°C)','Specific cond.\n(µS/cm)','pH','DO saturation\n(%)','Turbidity\n(FNU)','fDOM\n(RFU)','fDOM\n(QSU)']
    for name,label,color in [('FullHourly_shallow_qc.csv','~3 m','#0072b2'),('Profiles_35m_qc.csv','~35 m','#d55e00')]:
        frame=pd.read_csv(prepared/name,parse_dates=['TIMESTAMP'],na_values=['NAN'],float_precision='round_trip')
        frame=frame.loc[frame.TIMESTAMP.between(start,end)]
        for ax,(column,rule) in zip(axes,RULES.items()):
            y=pd.to_numeric(frame[column],errors='raise');valid=y.notna()
            ax.plot(frame.loc[valid,'TIMESTAMP'],y.loc[valid],'.',ms=.8,color=color,rasterized=True)
    for ax,label in zip(axes,labels):
        ax.set_ylabel(label,rotation=0,ha='right',va='center',fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=2));ax.tick_params(labelsize=8);ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlim(start,end)
    fig.legend(handles=[Line2D([],[],color=c,marker='.',linestyle='none',label=label) for c,label in [('#0072b2','~3 m'),('#d55e00','~35 m')]],ncol=4,loc='upper center',fontsize=9)
    fig.subplots_adjust(left=.17,right=.99,bottom=.07,top=.90,hspace=.20)
    save_figure(fig,figures,'Profiler_timeseries_QC');plt.close(fig)
    scada=pd.read_csv(prepared/'SCADA.csv',sep=';',parse_dates=['Time']).set_index('Time').loc[start:end]
    fig,axes=plt.subplots(2,1,figsize=(11.5,2.5),sharex=True)
    for ax,column in zip(axes,scada.columns):
        ax.plot(scada.index,scada[column],lw=.6,color='#0072b2');ax.set_ylabel(column.replace('SCADA - ','').replace(' (°C)','\n(°C)'),fontsize=9);ax.grid(alpha=.2)
    axes[-1].set_xlim(start,end);fig.tight_layout();save_figure(fig,figures,'SCADA_timeseries');plt.close(fig)
    clean,_,_=source_records(raw/'sensors/Weather_raw.csv',raw/'met_nordic')
    variables=list(clean['station'].columns)
    labels={'air_pressure_at_sea_level':'Pressure (mbar)','air_temperature_2m':'Temperature (°C)',
        'integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time':'Shortwave (W/m²)',
        'integral_of_surface_downwelling_longwave_flux_in_air_wrt_time':'Longwave (W/m²)',
        'precipitation_amount':'Rainfall (mm/h)','relative_humidity_2m':'Humidity (%)','wind_x':'Wind x (m/s)','wind_y':'Wind y (m/s)'}
    fig,axes=plt.subplots(8,1,figsize=(11.5,6.2),sharex=True)
    for source,color in [('grid','#d55e00'),('station','#0072b2')]:
        frame=clean[source].loc[start:end]
        for ax,variable in zip(axes,variables):ax.plot(frame.index,frame[variable],lw=.4,color=color,alpha=.75)
    for ax,variable in zip(axes,variables):
        ax.set_ylabel(labels[variable],rotation=0,ha='right',va='center',fontsize=8);ax.tick_params(labelsize=8);ax.yaxis.set_major_locator(MaxNLocator(nbins=2));ax.grid(axis='x',alpha=.2)
    axes[-1].set_xlim(start,end);fig.legend(handles=[Line2D([],[],color=c,label=l) for c,l in [('#0072b2','Station'),('#d55e00','MET Nordic, uncorrected')]],ncol=2,loc='upper center',fontsize=9)
    fig.subplots_adjust(left=.19,right=.99,bottom=.07,top=.94,hspace=.18)
    save_figure(fig,figures,'Weather_timeseries_alt_source');plt.close(fig)
    screening_table(prepared,tables)
    weather_table(raw,prepared,tables,out)
    seasonal_figure(prepared,figures)


def screening_table(prepared,tables):
    frame=pd.read_csv(prepared/'profiler_screening_summary.csv')
    frame=frame.loc[frame.scope.eq('study')]
    lines=[r'\begin{table}[tp]',r'\caption{Profiler screening ranges and recorded measurements at selected depths, 13 December 2021--10 February 2025.}\label{tab:profiler_screening}',r'\footnotesize',r'\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xllrr}',
        # Two-line headers (makecell) keep the four narrow columns narrow, so the Measurement
        # column is wide enough for every channel name on a single line.
        r'\toprule Measurement & \makecell[lb]{Sensor/physical\\range} & \makecell[lb]{Review\\range} & \makecell[rb]{Recorded,\\3~m} & \makecell[rb]{Recorded,\\35~m}\\\midrule']
    def bounds(lo,hi):
        if hi is None:return r'$\geq'+f'{lo:g}'+'$'
        return f'${lo:g}$--{hi:g}'
    for column,rule in RULES.items():
        # Summaries record channel names in the established raw schema.
        subset=frame.loc[frame.column.eq(column)]
        counts=[int(subset.loc[subset.series.eq(series),'recorded'].iloc[0]) for series in ['shallow','near35']]
        label = {'Sp Cond (microS_cm)': r'Specific conductance ($\upmu$S/cm)', 'DO (% Sat)': r'DO saturation (\%)'}.get(rule.label, tex(rule.label))
        lines.append(label+' & '+bounds(rule.sensor_min,rule.sensor_max)+' & '+bounds(rule.review_min,rule.review_max)+f' & {counts[0]:,} & {counts[1]:,}'+r'\\')
    if frame.excluded.eq(0).all() and frame.review_flagged.eq(0).all():
        note=r'Sensor-range exclusions and additional review flags are 0.00\% for every listed channel at both selected depths.'
    else:
        note='Exclusion and review percentages by series/channel are provided in the screening data.'
    lines.extend([r'\bottomrule',r'\end{tabularx}',r'\par\smallskip\scriptsize Recorded counts omit missing values. '+note+r" Specific conductance inherits failures of measured conductivity outside the wiped sensor's 0--100 mS/cm span. No fDOM RFU upper sensor limit is established by the specifications.",r'\end{table}'])
    atomic_text(tables/'profiler_screening.tex','\n'.join(lines)+'\n')


def weather_table(raw,prepared,tables,out):
    station=pd.read_csv(raw/'sensors/Weather_raw.csv',sep=';',decimal=',',parse_dates=['Time'],low_memory=False).set_index('Time').sort_index()
    grid=pd.read_csv(raw/'met_nordic/met_nordic_ltc_hourly.csv',parse_dates=['Time']).set_index('Time').sort_index()
    study=pd.date_range('2021-12-13','2025-02-10 12:00',freq='h')
    rows=[]
    for variable,(prefix,label,unit,_,_) in VARIABLES.items():
        column=next(c for c in station if prefix.lower() in c.lower())
        observed=pd.to_numeric(station[column],errors='coerce').replace([-9999,-999,-99.9,-99],np.nan)
        paired=pd.concat([observed.rename('station'),grid[variable].rename('grid')],axis=1,sort=True).dropna()
        error=(paired.grid-paired.station).to_numpy()
        if variable=='wind_direction_10m':
            error=(error+180)%360-180
            resultant=abs(np.mean(np.exp(1j*np.deg2rad(paired.station))))
            spread=np.rad2deg(np.sqrt(-2*np.log(np.clip(resultant,1e-15,1))))
        else:spread=paired.station.std(ddof=0)
        obs=observed.reindex(study);met=grid[variable].reindex(study)
        rows.append(dict(variable=variable,label=label,unit=unit,n_pairs=len(paired),study_fill=int((obs.isna()&met.notna()).sum()),
            bias=error.mean(),rmse=np.sqrt(np.square(error).mean()),standardized_bias=error.mean()/spread))
    pd.DataFrame(rows).to_csv(out/'weather_source_comparison.csv',index=False)
    lines=[r'\begin{table}[H]',r'\caption{Uncorrected MET Nordic minus station differences, 16 January 2020--15 August 2025. Bias/SD is in station-standard-deviation units. Gap-fill counts cover the study period.}\label{tab:weather_sources}',r'\small',r'\begin{tabularx}{\textwidth}{Xrrrrr}',r'\toprule Variable & Study fill & Paired hours & Bias & RMSE & Bias / SD\\\midrule']
    # Stored units are the data contract; only their printed form changes here.
    printed={'degC':r'$^\circ$C','W/m2':r'W/m$^2$','mm/hour':'mm/h','percentage points':r'\%','degrees':r'$^\circ$'}
    for row in rows:lines.append(minus(tex(row['label'])+' ('+printed.get(row['unit'],tex(row['unit']))+')'+f" & {row['study_fill']} & {row['n_pairs']} & {row['bias']:.3g} & {row['rmse']:.3g} & {row['standardized_bias']:.3g}"+r'\\'))
    lines.extend([r'\bottomrule',r'\end{tabularx}',r'\end{table}']);atomic_text(tables/'review_weather.tex','\n'.join(lines)+'\n')


# Calendar years are ordered, so they take an ordinal ramp: inferno sampled at 0.75, 0.58,
# 0.35 and 0.08, lightest (earliest year) to darkest (latest). Lightness is monotone, so the
# order of any two years reads without the legend. Each pair of years stays apart under normal
# vision and simulated protan/deutan/tritan vision (OKLab delta E >= 13), and the lightest
# step keeps 2.4:1 contrast on white.
YEAR_COLORS=['#f98e09','#d74b3f','#7f1e6c','#10092d']


def seasonal_figure(prepared,figures):
    import matplotlib.pyplot as plt
    records=substitute(pd.read_csv(prepared/'qualified_observations.csv',parse_dates=['TIMESTAMP']),'half')
    lab=records.loc[records.target.eq('Total coliforms 37°C (CFU/100mL)')].set_index('TIMESTAMP').value.sort_index()
    years=sorted(lab.index.year.unique())
    if len(years)!=len(YEAR_COLORS):
        raise ValueError(f'{len(years)} calendar years but the ordinal ramp has {len(YEAR_COLORS)} steps')
    fig,ax=plt.subplots(figsize=(10,3.4))
    for year,color in zip(years,YEAR_COLORS):
        observed=lab.loc[lab.index.year==year];earlier=lab.loc[lab.index.year<year]
        line=ax.plot(observed.index.dayofyear,observed,'o-',ms=3,lw=.9,color=color,label=str(year))[0]
        forecasts=[seasonal_window_mean(earlier.index.dayofyear.to_numpy(),earlier.index.hour.to_numpy(),earlier.to_numpy(),stamp.dayofyear,stamp.hour) for stamp in observed.index]
        ax.plot(observed.index.dayofyear,forecasts,'--',lw=1,color=line.get_color())
    ax.set_xlabel('Day of year');ax.set_ylabel('Total coliforms (CFU/100 mL)');ax.legend(ncol=4,fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();save_figure(fig,figures,'Seasonal_target_doy');plt.close(fig)
