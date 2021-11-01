import React from 'react';
import './AdditionalRowInfo.css';
import axios from "axios";

//Rendert die Erweiterte Ansicht, wenn der Nutzer diese Anfordert
class AdditionalRowInfo extends React.Component{
  constructor(props){
    super(props);
    this.state = {
      run_data : null,
      parameters : null,
      run_loaded : 0,
      data : this.props.data,
    };
    this.getRunID();
  }

  //fordert die Metadaten eines Durchlaufs für Input Parameters an,
  //wenn der Benutzer in die erweiterte Ansicht geht
  getRunID = () => {
    axios.get('http://localhost:' + process.env.REACT_APP_API_PORT + '/api/run/' + this.props.data.run_id)
      .then(res => {
        let without_params = res.data;
        let params = JSON.parse(res.data.parameters.slice(1,-1));
        delete without_params.parameters;

        this.setState({run_data : without_params, parameters : params, run_loaded : 1})
      })
      .catch(err => console.log(err));
  }

  render(){
    let detail_data = this.state.data;
    if(this.state.run_loaded === 1){
    }

    return(
      <div className="expand-container">
        <div className="expand">
          <h2>All Results</h2>
          <div className="expand-table-wrapper">
            <table>
              <thead>
                <tr className="expand-row">
                  <th>Keys</th>
                  <th>Values</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(detail_data).map((key,index) => {

                  return(
                    <tr key={index} className="expand-row">
                      <td className="expand-key">{key}: </td>
                      <td className="expand-value">{this.props.data[key]}</td> 
                    </tr>
                  )
                }
                )}
              </tbody>
            </table>
          </div>
        </div>
        <div className="expand">
          <h2>Input Parameters</h2>
          <div className="expand-table-wrapper">
            <table>
              <thead>
                <tr className="expand-row">
                  <th>Keys</th>
                  <th>Values</th>
                </tr>
              </thead>
              <tbody>
                {this.state.run_loaded === 1 &&
                  Object.keys(this.state.run_data).map((key,index) => {
                    return(
                      <tr key={index} className="expand-row">
                        <td className="expand-key">{key}: </td>
                        <td className="expand-value">{this.state.run_data[key]}</td> 
                      </tr>
                    )
                  }
                  )
                }
                {this.state.run_loaded === 1 &&
                    Object.keys(this.state.parameters).map((key,index) => {
                      return(
                        <tr key={index} className="expand-row">
                          <td className="expand-key">{key}: </td>
                          <td className="expand-key">
                            <div style={{overflow:"auto"}}>
                              {typeof this.state.parameters[key] === "boolean"
                                && this.state.parameters[key].toString()
                              }
                              {typeof this.state.parameters[key] === "object"
                                && (this.state.parameters[key].map((value,index) => {
                                  return(
                                    <div key={index}>
                                      {value}
                                    </div>
                                  )
                                }))
                              }
                              {typeof this.state.parameters[key] !== "boolean" &&
                                  typeof this.state.parameters[key] !== "object" &&
                                    this.state.parameters[key]
                              }
                                      
                                      
                            </div>
                          </td>
                        </tr>

                      )
                    })
                }
              </tbody>
            </table>
          </div>

        </div>
      </div>
    );
  }
}
export default AdditionalRowInfo;
