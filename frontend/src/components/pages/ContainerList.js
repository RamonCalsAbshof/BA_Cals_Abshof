import React, {useState, useEffect} from 'react'
import styled from 'styled-components';

const Wrapper = styled.div`
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(300px,1fr));
  row-gap: 20px;
  column-gap: 60px;
  justify-items: center;
  padding: 50px 0px;
`;

const ContainerBox = styled.div`
  padding:20px;
  width:300px;
  height:300px;
  border-radius:10px;
  box-shadow:5px 5px 15px 0px rgba(0,0,0,0.5);
  background-color: lightgrey;
`;

const testContainers = [1,2,3,4,5]

const ContainerList = () => {
  const [loading, setLoading] = useState(true)

  return (
      <Wrapper>
        {testContainers.map((value,index) => {
          return(
            <ContainerBox key={index}>
              Container: {value}
            </ContainerBox>
          )
        })}

      </Wrapper>
  )
}

export default ContainerList
