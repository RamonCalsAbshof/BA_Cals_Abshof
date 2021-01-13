import React from 'react';
import axios from "axios";
import './dynamic_form.css';
import ReactLoading from 'react-loading';
import AllFeaturesSelect from './all_features.js';

//Rendert die Ansicht, in der der Benutzer einen Algorithmus auswählen kann
class AlgorithmSelector extends React.Component{
  constructor(props){
    super(props);
    this.state = {
      algorithms:[],
      algorithm_index: null, 
      algorithm_set: 0,
      key:1,
    };
  }

  
  componentDidMount(){
    this.loadResults();
  }

  //prüft, ob ein Algorithm ausgewählt wurde und rendert die Parametereingabe
  selectAlgorithm = (e) => {
    if(e.target.value === "default"){
      this.setState({algorithm_set : 0})
    }else{
      this.setState({algorithm_index : e.target.value, algorithm_set : 1, key:this.state.key + 1})
    }
  }

  //fragt verfügbaren Algorithmen an
  loadResults = () => {
    axios.get("http://localhost:8000/algorithms")
      .then(res => this.setState({algorithms : res.data }))
      .catch(err => console.log(err));
  }

  render(){
    return(
      <div className="form-box">
        <h2>Algorithm Formpage</h2>
        <select name="algorithm_names" onChange={this.selectAlgorithm}>
          <option value="default"> select an algorithm </option>
          {this.state.algorithms.map((value,index) => {
            return( 
              <option key={index} value={index}>{value.algorithm_display_name}</option>
            )
          })}
        </select>
        {this.state.algorithm_set === 1 && <AlgorithmFormpage key={this.state.key} algorithm={this.state.algorithms[this.state.algorithm_index]} />}
      </div>
    );
  }
}

//Rendert die Ansicht, in der der Benutzer die Parameter für einen Algorithms eingeben kann 
class AlgorithmFormpage extends React.Component{
  constructor(props){
    super(props);
    let init_state = this.initState(this.props);
    this.state = {
      params : init_state,
      error : null,
      success : null,
      loading : false,
    }
  }

  //inititalisiert Listen-Obejekte im Zustand, falls Listen-Parameter vorhanden sind
  initState = (props) => {
    let algorithm = props.algorithm.params;
    let copy_state = {}
    for(let key of Object.keys(algorithm)){
      if(
        algorithm[key].type === "string_list"
        || algorithm[key].type === "number_list" 
        || algorithm[key].type === "select_list"
        || algorithm[key].type === "feature_list"
      ){
        let field_list = []
        let list_type = algorithm[key].type; 
        
        let min_fields = algorithm[key].min || 0;
        
        if(
          list_type === "string_list" 
          || list_type === "select_list"
          || list_type === "feature_list"
        ){
          for(let i = 0; i < min_fields; i++){
            field_list.push("")
          }
        }
        if(list_type === "number_list"){
          for(let i = 0; i < min_fields; i++){
            field_list.push(0)
          }
        }
        copy_state[key] = field_list;
      }
    }
    copy_state.algorithm_filename = props.algorithm.algorithm_filename;
    return copy_state;
  }

  //Verwaltet Zustandsänderungen für Zeichenketten
  handleChange = (e) => {
    if(e.target.value === ""){
      let copy_state = this.state.params;
      delete copy_state[e.target.name];
      this.setState({params : copy_state});
    }else{
      let copy_state = this.state.params; 
      copy_state[e.target.name] = e.target.value;
      this.setState({params : copy_state}); 
    }
  }

  //Verwaltet Zustandsänderungen für ganzzahlige Werte
  handleInteger = (e) => {
    e.preventDefault()
    if(e.target.value === ""){
      let copy_state = this.state.params
      delete copy_state[e.target.name];
      this.setState({params : copy_state});
    }else{
      //if e.target.value is number
      if(!isNaN(Number(e.target.value))){
        if(Number(e.target.value) % 1 === 0){
          let copy_state = this.state.params
          copy_state[e.target.name] = parseInt(e.target.value);
          this.setState({params : copy_state}); 
        }
      }
    }
  }


  //Verwaltet Zustandänderungen für ganzzahlige un dezimale Werte
  handleNumber = (e) => {
    e.preventDefault()
    if(e.target.value === ""){
      let copy_state = this.state.params
      delete copy_state[e.target.name];
      this.setState({params : copy_state});
    }else{
      //if e.target.value is number
      if(!isNaN(Number(e.target.value))){
        let copy_state = this.state.params
        copy_state[e.target.name] = parseFloat(e.target.value);
        this.setState({params : copy_state}); 
      }
    }
  }

  handleClick = (e) => {
    let copy_state = this.state.params
    copy_state[e.target.name] = e.target.value;

    this.setState({params : copy_state});
  }

  //Bringt die Werte des Zustands in die richtige Form für das Back-End  
  prepareSubmit = () => {
    let submit = {}
    let algorithm = this.props.algorithm.params
    for(let key of Object.keys(this.state.params)){
      if(key === "algorithm_filename"){
        submit[key] = this.state.params[key]
        continue;
      }
      let field_type = algorithm[key].type
      if(this.state.params[key] === ""){
        continue
      }
      if( 
        field_type === "number_list" 
        || field_type === "string_list" 
        || field_type === "select_list"
        || field_type === "feature_list"
      ){
        if(this.state.params[key].length === 0){
          continue
        }

        if( 
          field_type === "string_list" 
          || field_type === "select_list"
          || field_type === "feature_list"
        ){
          submit[key] = []
          for(let entry of this.state.params[key]){
            if(entry.length){
              submit[key].push(entry)
            }
          }
          continue
        }
      }
      if(field_type === "number_list"){
        submit[key] = []
        for(const entry of this.state.params[key]){
          submit[key].push(Number(entry));
        }
        continue
      }
      if(field_type === "number" || field_type === "integer"){
        submit[key] = Number(this.state.params[key]);
        continue
      }

      //Umwandlung von String zu Bool
      if(field_type === "bool"){
        if(this.state.params[key] === "true"){
          submit[key] = true
        }
        if(this.state.params[key] === "false"){
          submit[key] = false
        }
        continue
      }

      submit[key] = this.state.params[key]
    }
    return submit
  }

  //Sendet die Eingabewerte an das Back-End
  handleSubmit = (e) => {
    e.preventDefault()
    this.setState({loading : true});
    setTimeout(() => {
      let submit = this.prepareSubmit()
      axios.post("http://localhost:8000/submit/",submit)
        .then(res => {
          console.log(res);
          this.setState({loading : false, success : true, error : false});
        })
        .catch(error => {
          console.log(error);
          this.setState({loading : false, success : false, error : true});
        });
    },
      3000);
  }

  //Verwaltet Zustandsänderungen von Listen für dezimale Werte
  handleNumberList = (e, index) => {
    e.preventDefault()
    if(e.target.value === ""){
      return
    }else{
      if(!isNaN(Number(e.target.value))){
        let copy_state = this.state.params;
        copy_state[e.target.name][index] = parseFloat(e.target.value);
        this.setState({params : copy_state}); 
      }
    }
  }

  handleList = (e, index) => {
    e.preventDefault()
    let copy_state = this.state.params;
    copy_state[e.target.name][index] = e.target.value;
    this.setState({params : copy_state});
  }

  addItem = (key, list_type) => {
    let copy_state = this.state.params;
    if(list_type === "string_list" || list_type === "select_list" || list_type === "feature_list"){
      copy_state[key].push("");
    }
    if(list_type === "number_list"){
      copy_state[key].push(0);
    }
    this.setState({params : copy_state});
  }

  //Fügt einen Listenindex hinzu
  addItemMax = (key, max, list_type) => {
    let copy_state = this.state.params;
    if(list_type === "string_list" || list_type === "select_list" || list_type === "feature_list"){
      copy_state[key].push("");
    }
    if(list_type === "number_list"){
      copy_state[key].push(0);
    }
    if(copy_state[key].length === max){
      return;
    }
    this.setState({params : copy_state});
  }

  removeItem = (key) => {
    let copy_state = this.state.params;
    copy_state[key].pop()
    this.setState({params : copy_state});
  }

  //Verkleinert die Liste um einen Index
  removeItemMin = (key, min) => {
    let copy_state = this.state.params;
    if(copy_state[key].length === min){
      return;
    }
    copy_state[key].pop()
    //array.pop();
    this.setState({params : copy_state});

  }

  componentDidUpdate(){
    console.log(this.state);
  }

  componentDidMount(){
  }

  render(){
    //Die Parameter des Algorithmus 
    let algorithm = this.props.algorithm.params;

    let formelements = []
    let isRequired;
    let required_in_form = false;
    let has_default;
    //In der Schleife wird jeder Parameter aus der JSON-Datei einzeln durchlaufen
    for(let key of Object.keys(algorithm)){
      has_default = algorithm[key].hasOwnProperty("default")
      isRequired = false
      if((key === "algorithm_display_name") || (key === "algorithm_file_name")){
        continue;
      }
      

      //Überprüft, ob Parameter required ist
      if(algorithm[key].hasOwnProperty("required")){
        if(algorithm[key].required === "True"){
          isRequired = true;
          required_in_form = true;

        }
      }
      
      //STRING
      //Erstellt ein String-Eingabefeld für den Typ 'string'
      if(algorithm[key].type === "string"){
        let placeholder = null;
        if(has_default){
          placeholder = algorithm[key].default
        }else{
          placeholder = "no default"
        }
        formelements.push(
          <div>
            {isRequired && <b style={{color:"red"}}>*</b>}
            <label>{key}</label>
            <input type="text" name={key} 
              placeholder={placeholder != null && (placeholder)} 
              onChange={this.handleChange}
              required={isRequired}
            />
          </div>
        );
      }

      //INTEGER
      //Erstellt ein Integer-Eingabefeld für den Typ 'integer'
      if(algorithm[key].type === "integer"){
        let tooSmall = false;
        let tooBig = false;
          if(algorithm[key].hasOwnProperty("min")){
            if(this.state.params[key] < algorithm[key].min){
              tooSmall = true
            }
          }
          if(algorithm[key].hasOwnProperty("max")){
            if(this.state.params[key] > algorithm[key].max){
              tooBig = true
            }
          }
        formelements.push(
          <div>
            {isRequired && <span style={{color:"red"}}>*</span>}
            <label>{key}</label>
            <input type="number" name={key} 
              value={this.state.params[key] || ''}
              placeholder={has_default ? algorithm[key].default : "no default" } 
              onChange={this.handleInteger}
              min={algorithm[key].min || ""}
              max={algorithm[key].max || ""}
              required={isRequired}
            />
            {tooSmall && <span style={{color:"red"}}> Select bigger number (min value: {algorithm[key].min})</span>}
            {tooBig && <span style={{color:"red"}}> Select smaller number (max value: {algorithm[key].max})</span>}
          </div>
        );
      }

      //NUMERISCH ALLGEMEIN
      //Erstellt ein Eingabefeld für den Typ 'number'
      if(algorithm[key].type === "number"){
        let tooSmall = false;
        let tooBig = false;
          if(algorithm[key].hasOwnProperty("min")){
            if(this.state.params[key] < algorithm[key].min){
              tooSmall = true
            }
          }
          if(algorithm[key].hasOwnProperty("max")){
            if(this.state.params[key] > algorithm[key].max){
              tooBig = true
            }
          }
        formelements.push(
          <div>
            {isRequired && <span style={{color:"red"}}>*</span>}
            <label>{key}</label>
            <input type="number" name={key} 
              placeholder={has_default ? algorithm[key].default : "no default" } 
              value={this.state.params[key] || ''}
              onChange={this.handleNumber}
              min={algorithm[key].min || undefined}
              max={algorithm[key].max || undefined}
              required={isRequired}
              step={algorithm[key].stepSize || "0.01"}
            />
            {tooSmall && <span style={{color:"red"}}> Select bigger number (min value: {algorithm[key].min})</span>}
            {tooBig && <span style={{color:"red"}}> Select smaller number (max value: {algorithm[key].max})</span>}
          </div>
        );
      }

      //STRING LIST oder NUMBER LIST
      //Erstellt ein oder mehrere Eingabefelder für den Typ 'string_list' oder 'number_list'
      if(algorithm[key].type === "string_list" || algorithm[key].type === "number_list"){
        let list_type = algorithm[key].type;
        let min_len = algorithm[key].min || 0;
        let max_len = algorithm[key].max || 100;
        formelements.push(
          <div>
            {this.state.params[key].map((value,index) => {
              return(
                <div key={index} className="list-param">
                  {index < min_len
                    ? <span style={{color:"red"}}>*</span>
                    : <span style={{visibility:"hidden"}}>*</span>
                  }

                  <label>{key+(index+1)}</label>
                  {list_type === "string_list" &&
                  <input 
                    type="text" 
                    name={key} 
                    value={this.state.params[key][index]} 
                    onChange={(event) => this.handleList(event, index)}
                    required={index < min_len}
                  />
                  }
                  {list_type === "number_list" &&
                  <input
                    type="number"
                    name={key}
                    value={this.state.params[key][index]}
                    onChange={(event) => this.handleNumberList(event, index)}
                    required={index < min_len}
                  />
                  }
                </div>
              )
            })}
            <button type="button" onClick={() => this.addItemMax(key,max_len,list_type)}>+ Add {key}</button>
            <button type="button" onClick={() => this.removeItemMin(key,min_len)}>- Remove {key}</button>
          </div>
        );
      }

      //FEATURE
      //Erstellt ein Optionen-Feld für den Typ 'feature'
      if(algorithm[key].type === "feature"){
        formelements.push(
          <div>
            {isRequired && <span style={{color:"red"}}>*</span>}
            <label>{key}</label>
            <select 
              name={key} 
              onChange={this.handleChange}
              required={isRequired}
            >
              {has_default 
                ? <option 
                  value={algorithm[key].default} 
                  key="default_elem">{algorithm[key].default}</option>
                  : <option
                    value=""
                    key="default_elem">--Please Select--</option>
              }
              <AllFeaturesSelect/>
            </select>
          </div>
        );
      }


      //FEATURE LIST
      //Erstellt ein oder mehrere Eingabefelder für den Typ 'feature_list'
      if(algorithm[key].type === "feature_list"){
        let list_type = "select_list"
        let min_len = algorithm[key].min || 0;
        let max_len = algorithm[key].max || 100;
        formelements.push(
          <div>
            {this.state.params[key].map((value,index) => {
              return(
                <div key={index} className="list-param">
                  {index < min_len
                    ? <span style={{color:"red"}}>*</span>
                    : <span style={{visibility:"hidden"}}>*</span>
                  }
                  <label>{key+(index+1)}</label>
                  <select name={key} 
                    onChange={(event) => this.handleList(event, index)}
                    required={index < min_len}
                  >
                    <option 
                      value=""
                      key="default_elem"
                    >
                      --Please Select--
                    </option>
                    <AllFeaturesSelect/>
                  </select>
                </div>
              )})
            }
            <button type="button" onClick={() => this.addItemMax(key,max_len,list_type)}>+ Add {key}</button>
            <button type="button" onClick={() => this.removeItemMin(key,min_len)}>- Remove {key}</button>
          </div>
        );
      }


      //SELECT LIST
      //Erstellt ein oder mehrere Eingabefelder für den Typ 'select_list'
      if(algorithm[key].type === "select_list"){
        let options_copy = algorithm[key].options.slice()
        let list_type = algorithm[key].type
        let min_len = algorithm[key].min || 0;
        let max_len = algorithm[key].max || 100;
        formelements.push(
          <div>
            {this.state.params[key].map((value,index) => {
              return(
                <div key={index} className="list-param">
                  {index < min_len
                    ? <span style={{color:"red"}}>*</span>
                    : <span style={{visibility:"hidden"}}>*</span>
                  }
                  <label>{key+(index+1)}</label>
                  <select 
                    name={key} 
                    onChange={(event) => this.handleList(event, index)}
                    required={index < min_len}
                  >
                    <option 
                      value=""
                      key="default_elem"
                    >
                      --Please Select--
                    </option>
                    {options_copy.map((value, index) => {
                      return(
                        <option value={value} key={index}>{value}</option>
                      )
                    })}
                  </select>
                </div>
              )})
            }
            <button type="button" onClick={() => this.addItemMax(key,max_len,list_type)}>+ Add {key}</button>
            <button type="button" onClick={() => this.removeItemMin(key,min_len)}>- Remove {key}</button>
          </div>
        );
      }

      //SELECT 
      //Erstellt ein Eingbaefeld für den Typ 'select'
      if(algorithm[key].type === "select"){
        let options_copy;
        let default_elem;
        if(algorithm[key].type === "select"){
          options_copy = algorithm[key].options.slice()
        }

        if(algorithm[key].type === "dataset"){
          options_copy = [
            "restatements-negative.csv",
            "restatements-positive.csv",
            "restatements.csv",
            "restatements_audit_analytics.csv",
            "testdatei.csv",
            "testdateimitlabels.csv"]
        }

        if(has_default){
          default_elem = algorithm[key].default
          options_copy.splice(options_copy.indexOf(default_elem), 1);
        }

        formelements.push(
          <div>
            {isRequired && <span style={{color:"red"}}>*</span>}
            <label>{key}</label>
            <select 
              name={key} 
              onChange={this.handleChange}
              required={isRequired}
            >
              {has_default 
                ? <option 
                  value={algorithm[key].default} 
                  key="default_elem">{algorithm[key].default}</option>
                  : <option
                    value=""
                    key="default_elem">--Please Select--</option>
              }
              {options_copy.map((value, index) => {
                return(
                  <option value={value} key={index}>{value}</option>
                )
              })}
            </select>
          </div>
        );
      }

      //BOOL
      //Erstellt ein Eingabefeld für den Typ 'bool'
      if(algorithm[key].type === "bool"){
        formelements.push(
          <div>
            {isRequired && <span style={{color:"red"}}>*</span>}
            <label>{key}</label>
            <select 
              name={key} 
              onChange={this.handleChange}
              required={isRequired}
            >
              {has_default 
                ? (
                algorithm[key].default === "True" 
                  ?(
                    <>
                      <option value="true">True</option>
                      <option value="false">False</option>
                    </>
                  ) 
                  :(algorithm[key].default === "False" 
                    &&(
                      <>
                        <option value="false">False</option>
                        <option value="true">True</option>
                      </>
                    )
                  ) 
                )
                :(
                  <>
                    <option value="">--Please Select--</option>
                    <option value="false">False</option>
                    <option value="true">True</option>
                  </>
                )
              }
            </select>
          </div>
        );
      }
    }

    return(
      <div className="param-box">
        {required_in_form && 
          <div style={{margin:"10px"}}>
            <span style={{color:"red"}}>*</span>
            <span> = required parameter</span>
          </div>
        }
        <form onSubmit={this.handleSubmit}>
        {formelements.map((element, index) => {
          return(
            <React.Fragment key={index}>
              {element}
            <br/>
            </React.Fragment>
          );
        })}
          {!this.state.loading 
            ? <input type="submit" value="Submit"/>
            : (
            <div>
              <ReactLoading type={"spin"} color={"blue"}/>
            </div>
            )
          }
          {this.state.error &&
            <span className="error">Something went wrong, please try again</span>
          }
          {this.state.success &&
            <span className="success">Submitted</span>
          }
        </form>
      </div>
    );
  }
}

export default AlgorithmSelector;
