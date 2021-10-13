import React from 'react';
import axios from "axios";
import DataTable from 'react-data-table-component';
//DataTable von https://www.npmjs.com/package/react-data-table-component#datatable-properties
import './dynamic_results.css';
import AdditionalRowInfo from './additional_row_info.js';

//Fordert alle Ergebnisdaten der Algorithmen an 
//Wenn alle Daten vom Back-End angekommen sind, wird die Tabelle mit den Ergebnisdaten gerendert
class DynamicResults extends React.Component{
  constructor(props){
    super(props);
    this.state = {
      table_data:[],
      dataIsLoaded:false,
    };
    this.loadTabledata();
  }

  componentDidMount(){
  }

  componentDidUpdate(){
  }

  loadTabledata = () => {
    axios.get('http://localhost:' + process.env.REACT_APP_API_PORT + '/api/tabledata')
      .then(res => this.setState({table_data : res.data.table_data, dataIsLoaded : true }))
      .catch(err => console.log(err));
  }

  render(){
    //Wenn alle Ergebnisdaten vorhanden sind, wird die Ergebnistabelle gerendert
    return(this.state.dataIsLoaded &&
          <ResultTable 
            table_data = {this.state.table_data}
            />
    );
  }
}

//Rendert die Ergebnistabelle
class ResultTable extends React.Component{
  constructor(props){
    super(props);
    //Die Attribute in header_spec werden direkt in der Ergebnistabelle auf der Benutzeroberfläche angezeigt
    const header_spec =[
      {selector:"algorithm_display_name",name:"Algorithm", sortable:true, wrap:true, compact:true, minWidth:"150px", left:true},
      {selector:"precision",name:"Precision", sortable:true, wrap:true, compact:true, right:true},
      {selector:"recall",name:"Recall", sortable:true, wrap:true, compact:true, right:true},
      {selector:"f1_score",name:"F1-Score", sortable:true, wrap:true, compact:true, right:true},
      {selector:"accuracy",name:"Accuracy", sortable:true, wrap:true, compact:true, right:true},
      {selector:"label",name:"Label", sortable:true, wrap:true, compact:true, center:true,minWidth:"100px"},
      {selector:"finished_date",name:"Finished Date", sortable:true, wrap:true, compact:true, center:true}]

    this.state = {
      column_names : header_spec,
      table_data : this.props.table_data,
      filtered_data : this.props.table_data,
      label_filter : "",
      algorithm_display_name_filter : ""
    };
  }

  componentDidMount(){
  }

  componentDidUpdate(){
  }

  //Ändert den Zustand, wenn der Benutzer ein Label in das Suchfeld eingibt
  setLabelFilter = (e) => {
    this.setState({label_filter : e.target.value})
  }
  
  //Ändert den Zustand, wenn der Benutzer einen Algorithmusnamen in das Suchfeld eingibt
  setAlgorithmFilter = (e) => {
    this.setState({algorithm_display_name_filter : e.target.value})
  }

  //Filtert die Tabelleneinträge auf der Benutzeroberfläche nach dem Label und dem Algorithmusnamen
  filterPipeline = () => {
    let filtered_table_data = this.state.table_data.filter(row => (
      row.label.toLowerCase().includes(this.state.label_filter.toLowerCase())
      && row.algorithm_display_name.toLowerCase().includes(this.state.algorithm_display_name_filter.toLowerCase())
    ))
    return filtered_table_data;
  }

  render(){
    //Enthält die Tabelleneinträge, die den Suchkriterien entsprechen
    //Wenn keine Suchkriterien angegeben sind, sind alle Daten enthalten
    let filtered_data = this.filterPipeline();
    return(
      <div className="table-wrapper">
        <input type="text" placeholder="Algorithm Filter" name="algorithm_filter" onChange={this.setAlgorithmFilter}/>
        <input type="text" placeholder="Label Filter" name="label_filter" onChange={this.setLabelFilter}/>

        <DataTable 
          className="table"
          title="Result Table" 
          columns={this.state.column_names} 
          data={filtered_data}  
          pagination
          expandableRows
          dense
          expandableRowsComponent={<AdditionalRowInfo/>}
          striped
        />
      </div>
    );
  }
}

export default DynamicResults;
