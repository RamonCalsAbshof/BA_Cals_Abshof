import React, { useEffect } from 'react';
import styled from 'styled-components';
import video from './LandingPageVideo.mp4'
import { Link } from 'react-router-dom';

const datasets = ['Eikon', 'Audit Analytics']

const algorithms = [
  'Support Vector Classification Positive/Negative (Audit Analytics)',
  'Support Vector Classification (TR Eikon) - Featureboost',
  'AdaBoost Classification (TR Eikon) - Featureboost',
  'Support Vector Classification Positive (Audit Analytics)',
  'Support Vector Classification (TR Eikon)',
  'AdaBoost Classification (TR Eikon)',
  'Support Vector Classification Negative (Audit Analytics)',
  'Support Vector Classification Positive (Audit Analytics)- Featureboost',
  'Isolation Forest (TR Eikon) - Featureboost',
  'Random Forest Classification (TR Eikon)',
  'KNeighbors Classification (TR Eikon) - Featureboost',
  'Outlier Detector',
  'Decision Tree Classification (TR Eikon) - Featureboost',
  'Support Vector Classification Positive/Negative (Audit Analytics)- Featureboost',
  'Support Vector Classification Negative (Audit Analytics)- Featureboost',
  'KNeighbors Classification (TR Eikon)',
  'Random Forest Classification (TR Eikon) - Featureboost',
  'Isolation Forest (TR Eikon)',
  'Decision Tree Classification (TR Eikon)'
]

const Wrapper = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: ${props => props.justifyContent || "center"};
  align-items: ${props => props.alignItems || "center"};
  padding-bottom: ${props => props.paddingBottom || "150px"};
  gap:20px;
`;

const TextWrapper = styled.div`
  display:flex;
  flex-direction: column;
  justify-content: center;
  width: 500px;
  padding-bottom: 50px;
`;

const PreviewWrapper = styled.div`
  max-height:600px;
  max-width:800px;
  border-radius:10px;
  box-shadow: 5px 6px 15px -2px rgba(0,0,0,0.5);
`;

const Video = styled.video`
  max-height:600px;
  max-width:800px;
  border-radius:10px;
  box-shadow: 5px 6px 14px -2px grey;
`;

const Header = styled.h1`
  font-size: 34px;
  font-weight: 500;
  padding: 10px 0px;
`;

const SubHeader = styled.p`
  font-size: 16px;
  color: rgba(0, 0, 0, 0.5);
  padding: 10px 0px;
`;

const ButtonWrapper = styled.div`
  padding: 10px 0px;
`;

const CalcButton = styled.button`
  border: none;
  border-radius: 20px;
  padding: 10px 15px;
  font-size: 16px;
  display: inline-block;
  width: fit-content;
  background-color: #fff;
  box-shadow: 4px 5px 15px -2px rgba(0,0,0,.3);

  &:hover {
    background-color: #0c3f93; 
    cursor: pointer;
    color: #eee;
    transition: all .4s ease;
  }
`;

const InfoCard = styled.div`
  border: 1px solid #ddd;
  border-radius: 15px;
  width: fit-content;
  padding: 7px 15px;
  margin-bottom: 5px;
`;

const InfoCardHeader = styled.p`
  padding-bottom: 20px;
  font-size: 16px;
`;

const LandingPage = () => {
  return(
    <React.Fragment>
      <Wrapper>
        <TextWrapper>
          <Header>
            Analysieren Sie Bilanzen börsennotierter amerikanischer Unternehmen.
          </Header> 
          <SubHeader>
            Analysieren Sie aus den Bilanzen amerikanischer börsennotierter 
            Unternehmen gewonnene Finanzdaten mithilfe fortschrittlicher Algorithmen 
            aus dem Bereich des maschinellen Lernens.
          </SubHeader>
          <ButtonWrapper>
            <Link to="/new_calc">
              <CalcButton>
                Analyse Starten
              </CalcButton>
            </Link>
          </ButtonWrapper>
        </TextWrapper>
        <Video loop autoPlay muted src={video}>
          <source src={video} type="video/mp4"></source>
        </Video>
      </Wrapper>
      <Wrapper paddingBottom="50px">
        <h2 style={{
          textAlign:"center",
          fontSize:"30px",
          fontWeight:"500",
          letterSpacing:"0.5px"
        }}>
          Wählen Sie einen der verfügbaren Datensätze,
          einen passenden Algorithmus, und starten Sie Ihre Analyse.
        </h2>
      </Wrapper>
      <Wrapper alignItems="flex-start" justifyContent="center">
        <div style={{
          width:"30%", 
          display:"flex", 
          alignItems:"flex-end",
          flexDirection:"column"}}>
          <InfoCardHeader>Datensätze</InfoCardHeader>

          {datasets.map((value,index) => {
            return (
              <InfoCard key={index}>
                {value}
              </InfoCard>
            )
          })}
        </div>
        <div style={{border:"1px dotted grey", height:"300px"}}></div>
        <div style={{width:"30%"}}>
          <InfoCardHeader>Algorithmen</InfoCardHeader>
          {algorithms.map((value,index) => {
            return (
              <InfoCard key={index}>
                {value}
              </InfoCard>
            )
          })}
        </div>

      </Wrapper>
    </React.Fragment>
  ) 
}

export default LandingPage;