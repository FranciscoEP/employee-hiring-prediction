import React, { useEffect } from 'react';
import { useRouter } from 'next/router';
import Success from '../components/Success';
import Reject from '../components/Reject';

const Results = () => {
    const router = useRouter();
    // useEffect(() => {
    //     if (!router?.asPath.includes('0') || !router?.asPath.includes('1'))
    //         router.push('/');
    // }, [router]);

    return router?.asPath.includes('1') ? <Success /> : <Reject />;
};

export default Results;
