"""
data.py – All input, output, and parameter dataclasses for the FraNchEstYN model.

Translated from:
  data/data.cs
  data/simulationUnit.cs
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List


# ---------------------------------------------------------------------------
# Input classes
# ---------------------------------------------------------------------------

@dataclass
class InputsHourly:
    """Hourly input record for the disease model."""
    date: datetime = datetime.min
    air_temperature: float = 0.0        # °C
    precipitation: float = 0.0          # mm
    relative_humidity: float = 0.0      # %
    leaf_wetness: float = 0.0           # 0 or 1
    rad: float = 0.0                    # MJ m⁻² h⁻¹
    dis_ideotype_potential_rate: float = 0.0  # unitless
    latitude: float = 0.0               # degrees
    date_treatment_last: datetime = datetime.min


@dataclass
class InputsDaily:
    """Daily input record (assembled from 24 hourly records each day)."""
    date: datetime = datetime.min
    tmax: float = 0.0                   # °C
    tmin: float = 0.0                   # °C
    rad: float = 0.0                    # MJ m⁻² d⁻¹
    rhx: float = 0.0                    # % (max RH)
    rhn: float = 0.0                    # % (min RH)
    precipitation: float = 0.0          # mm
    leaf_wetness: float = 0.0           # hours
    dew_point: float = 0.0              # °C
    latitude: float = 0.0               # degrees
    date_treatment_last: datetime = datetime.min
    crop_model_data: "CropModelData" = None  # populated from external crop model CSV


# ---------------------------------------------------------------------------
# Crop model data (external model output fed into Franchestyn)
# ---------------------------------------------------------------------------

@dataclass
class CropModelData:
    """Output from an external crop model, used as input to damage mechanisms.

    Keys are datetime.date objects (year, month, day).
    """
    f_int: Dict[date, float] = field(default_factory=dict)          # fraction 0-1: light interception
    yield_: Dict[date, float] = field(default_factory=dict)         # kg ha⁻¹: dynamic crop yield
    agb: Dict[date, float] = field(default_factory=dict)            # kg ha⁻¹: above-ground biomass
    cycle_percentage: Dict[date, float] = field(default_factory=dict)  # %: crop cycle completion
    # NOTE: gdd is NOT in the original C# cropModelData. It was added in this Python
    # rewrite to fix a bug where growingDegreeDays was always 0 in the C# code.
    # Populated from SIMPLACE's TSUM column via reference_reader.read_crop_model_data().
    gdd: Dict[date, float] = field(default_factory=dict)            # °C·day: growing degree days (TSUM from SIMPLACE)


# ---------------------------------------------------------------------------
# Output classes
# ---------------------------------------------------------------------------

@dataclass
class TissueState:
    """State of a single tissue cohort (SEIR disease model)."""
    latent_state: float = 0.0           # fraction currently in latent phase
    sporulating_state: float = 0.0      # fraction currently sporulating
    dead_state: float = 0.0             # fraction dead
    latent_counter: float = 0.0         # progress counter in latent phase (→ 1 = transitions to sporulating)
    sporulating_counter: float = 0.0    # progress counter in sporulating phase (→ 1 = transitions to dead)


@dataclass
class DamageMechanisms:
    """Damage outputs (one per day) that feed into crop loss calculations."""
    light_stealers: float = 0.0         # fraction: reduction in light interception
    senescence_accelerators: float = 0.0  # fraction: acceleration of tissue aging
    rue_reducers: float = 0.0           # fraction: reduction in radiation use efficiency
    assimilate_sappers: float = 0.0     # kg ha⁻¹: assimilate loss from disease


@dataclass
class DiseaseOutputs:
    """All disease state variables output each day."""
    counter_dry: float = 0.0            # hours since last wet period
    temp_function: float = 0.0          # temperature suitability (0-1)
    rh_function: float = 0.0            # RH suitability (0-1)
    hydro_thermal_time_state: float = 0.0   # accumulated hydro-thermal time (hydro-degree days)
    hydro_thermal_time_rate: float = 0.0    # daily hydro-thermal time rate
    hydro_thermal_time_infection: float = 0.0  # infection accumulation

    sporulation_efficiency: float = 0.0  # %

    tmax_daily: float = 0.0             # °C
    tmin_daily: float = 0.0             # °C
    rain_daily: float = 0.0             # mm
    rhmax_daily: float = 0.0            # %
    rhmin_daily: float = 0.0            # %
    lw_daily: float = 0.0               # % (leaf wetness fraction of day)

    # Tissue tracking is handled as a list of TissueState in disease_model.py
    # Summary aggregates written here each day:
    tissue_state: TissueState = field(default_factory=TissueState)
    latent_sum: float = 0.0             # % cumulative latent
    sporulating_sum: float = 0.0        # % cumulative sporulating
    dead_sum: float = 0.0               # % cumulative dead
    affected_sum: float = 0.0           # % cumulative affected
    disease_severity: float = 0.0       # % disease severity (sporulating + dead)
    susceptible_fraction: float = 1.0   # fraction of healthy susceptible tissue

    damage_mechanisms: DamageMechanisms = field(default_factory=DamageMechanisms)

    first_seasonal_infection: datetime = field(default_factory=lambda: datetime(1900, 1, 1))
    cycle_percentage_first_infection: float = 0.0
    is_primary_inoculum_started: bool = False
    outer_inoculum: float = 0.0


@dataclass
class CropOutputs:
    """Crop growth state variables output each day."""
    growing_season: int = 0             # season identifier
    day_after_sowing: int = 0           # days since sowing
    pheno_code: float = 0.0             # phenological code (1 = vegetative, 2 = reproductive)
    growing_degree_days: float = 0.0    # °C·day accumulated GDD
    agb_attainable: float = 0.0         # kg ha⁻¹ attainable AGB
    agb_actual: float = 0.0             # kg ha⁻¹ actual AGB
    yield_attainable: float = 0.0       # kg ha⁻¹ attainable yield
    yield_actual: float = 0.0           # kg ha⁻¹ actual yield
    light_interception_attainable: float = 0.0  # fraction
    light_interception_actual: float = 0.0       # fraction
    cycle_completion_percentage: float = 0.0     # %
    senescence_started: bool = False
    f_int_peak: float = 0.0


@dataclass
class FungicideOutputs:
    """Fungicide state variables output each day."""
    concentration_factor: float = 0.0   # fraction - fungicide concentration
    tenacity_function: float = 0.0      # unitless - fungicide tenacity function
    tenacity: float = 0.0               # fraction - fungicide tenacity (potential efficacy after wash-off)     
    actual_degradation: float = 0.0     # fraction (1 = not degraded)
    potential_degradation: float = 0.0  # fraction
    efficacy: float = 0.0               # fraction


@dataclass
class Outputs:
    """Top-level daily output bundle."""
    disease: DiseaseOutputs = field(default_factory=DiseaseOutputs)
    crop: CropOutputs = field(default_factory=CropOutputs)
    inputs_daily: InputsDaily = field(default_factory=InputsDaily)
    fungicide: FungicideOutputs = field(default_factory=FungicideOutputs)


# ---------------------------------------------------------------------------
# Parameter classes
# ---------------------------------------------------------------------------

@dataclass
class ParDisease:
    """Disease model parameters."""
    outer_inoculum_max: float = 0.0             # unitless – initial inoculum level
    outer_inoculum_shape_release: float = 0.0   # 0/1/2 – release shape
    outer_inoculum_shape_parameter: float = 0.0 # empirical parameter for shape 1 or 2
    pathogen_spread: float = 0.0                # dispersal potential
    wetness_duration_optimum: float = 0.0       # hours – optimal leaf wetness
    wetness_duration_minimum: float = 0.0       # hours – minimum required wetness
    dry_critical_interruption: float = 0.0      # hours – dry spell that disrupts infection
    rain50_detachment: float = 0.0              # mm – rain needed for spore detachment
    cycle_percentage_onset: float = 0.0         # % – crop cycle % when disease can start
    tmin: float = 0.0                           # °C – minimum temperature for pathogen
    topt: float = 0.0                           # °C – optimum temperature
    tmax: float = 0.0                           # °C – maximum temperature
    relative_humidity_critical: float = 0.0     # % – minimum RH for infection
    relative_humidity_not_limiting: float = 0.0 # % – RH above which it is not limiting
    hydro_thermal_time_onset: float = 0.0       # hydro-degree days – onset threshold
    latency_duration: float = 0.0               # days – latency period duration
    sporulation_duration: float = 0.0           # days – sporulation duration
    light_stealer_damage: float = 0.0           # unitless – lesion size representation
    rue_reducer_damage: float = 0.0             # fraction – RUE reduction
    senescence_accelerator_damage: float = 0.0  # fraction – senescence acceleration rate
    assimilate_sappers_damage: float = 0.0      # kg ha⁻¹ – max assimilate loss
    is_splash_borne: bool = False               # True if spore release requires rain


@dataclass
class ParCrop:
    """Crop model parameters."""
    tbase_crop: float = 0.0             # °C – base temperature for crop growth
    topt_crop: float = 0.0              # °C – optimum temperature
    tmax_crop: float = 0.0              # °C – maximum temperature
    cycle_length: float = 0.0           # degree-days – total crop cycle length
    flowering_start: float = 0.0        # % – crop cycle % when flowering starts
    half_int_growth: float = 0.0        # % – cycle % when light interception = 0.5 (growth phase)
    half_int_senescence: float = 0.0    # % – cycle % when light interception = 0.5 (senescence phase)
    slope_growth: float = 0.0           # unitless – slope of growth light interception curve
    slope_senescence: float = 0.0       # unitless – slope of senescence curve
    radiation_use_efficiency: float = 0.0  # g MJ⁻¹
    partitioning_maximum: float = 0.0   # unitless – max yield partitioning fraction
    varietal_resistance: float = 0.0    # 0–1 – varietal resistance to pathogen


@dataclass
class ParFungicide:
    """Fungicide model parameters."""
    a_shape_parameter: float = 0.0      # empirical degradation shape A
    b_shape_parameter: float = 0.0      # empirical degradation shape B
    degradation_rate: float = 0.0       # first-order degradation rate
    initial_dose: float = 0.0           # fraction – initial dose (1 = max dose)
    initial_efficacy: float = 0.0       # fraction – initial efficacy (1 = max)
    tenacity_factor: float = 0.0        # rain-driven degradation empirical factor


@dataclass
class Parameters:
    """Container for all three parameter groups."""
    par_disease: ParDisease = field(default_factory=ParDisease)
    par_crop: ParCrop = field(default_factory=ParCrop)
    par_fungicide: ParFungicide = field(default_factory=ParFungicide)


@dataclass
class Parameter:
    """Represents a single parameter definition (from CSV) with bounds for calibration.
        Originally defined in parameterReader.cs, but used across the model and optimizer."""
    minimum: float = 0.0                # Lower bound (e.g., calibration search space)
    maximum: float = 0.0                # Upper bound (e.g., calibration search space)
    value: float = 0.0                  # Current numeric value (e.g., for simulation or optimization)
    calibration: str = ""               # calibration subset tag (e.g., "included", "excluded")
    param_class: str = ""               # class/category of the parameter (e.g., "disease", "crop")
    value_bool: bool = False            # For boolean parameters, the value is stored here instead of 'value'
    is_boolean: bool = False            # True if this parameter is boolean (e.g., "IsSplashBorne"), in which case 'value_bool' is used instead of 'value'


# ---------------------------------------------------------------------------
# Simulation unit (site × variety × year setup)
# ---------------------------------------------------------------------------

@dataclass
class FungicideTreatmentSchedule:
    """Holds all scheduled fungicide treatment dates for a simulation unit."""
    treatments: List[datetime] = field(default_factory=list)

    def add_treatment(self, treatment_date: datetime) -> None:
        self.treatments.append(treatment_date)


@dataclass
class ReferenceData:
    """Observed / reference data for validation (field measurements)."""
    date_fint: Dict[date, float] = field(default_factory=dict)
    date_agb: Dict[date, float] = field(default_factory=dict)
    date_yield_actual: Dict[date, float] = field(default_factory=dict)
    date_yield_attainable: Dict[date, float] = field(default_factory=dict)
    # disease_date_disease_sev[pathogen][date] = severity
    disease_date_disease_sev: Dict[str, Dict[date, float]] = field(default_factory=dict)


@dataclass
class SimulationUnit:
    """Everything needed to run one site × variety combination."""
    site: str = ""                  # site identifier (e.g., location name)
    crop: str = ""                  # crop identifier (e.g., crop name)
    latitude: float = 0.0           # degrees - site latitude
    longitude: float = 0.0          # degrees - site longitude
    variety: str = ""               # variety identifier (e.g., variety name)
    pathogen: str = ""              # pathogen identifier
    variety_resistance: float = 0.0 # 0–1 – varietal resistance to pathogen

    year_sowing_doy: Dict[int, int] = field(default_factory=dict)           # Maps simulation year → sowing day-of-year

    reference_data: ReferenceData = field(default_factory=ReferenceData)    # observed or reference data for validation 
    fungicide_treatment_schedule: FungicideTreatmentSchedule = field(       # fungicide treatment scheduling
        default_factory=FungicideTreatmentSchedule
    )
