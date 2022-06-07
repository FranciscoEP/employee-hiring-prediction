import {
    Box,
    Grid,
    Stack,
    Select,
    Heading,
    Text,
    Container,
    Input,
    Button,
    SimpleGrid,
    FormControl,
    FormLabel,
} from '@chakra-ui/react';

import { useState } from 'react';
import { useRouter } from 'next/router';
import Hero from './Hero';

export function Form() {
    const router = useRouter();
    const [formValues, setFormValues] = useState({
        gender: 'Female',
        bachelor_score: 0.0,
        master_score: 0.0,
        experience_test: 0.0,
        work_experience: 'Yes',
    });

    const handleChange = (e) => {
        setFormValues((prevState) => ({
            ...prevState,
            [e.target.name]: e.target.value,
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const finalValues = {
            gender: formValues.gender === 'Male' ? 1 : 0,
            bachelor_score: formValues.bachelor_score,
            master_score: formValues.master_score,
            experience_test: formValues.experience_test,
            work_experience: formValues.work_experience === 'Yes' ? 1 : 0,
        };
        try {
            const response = await fetch('http://localhost:8000/prediction', {
                method: 'POST',
                credentials: 'include',
                mode: 'cors',
                body: JSON.stringify(finalValues),
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (response.ok) {
                const json = await response.json();
                return router.push(`/results?${json.data.prediction}`);
            }
            throw Error(`Request rejected with status ${response.status}`);
        } catch (error) {
            // setLoading(false);
            // setPasswordError(true);
        }
    };

    return (
        <Container
            as={SimpleGrid}
            maxW="7xl"
            columns={{ base: 1, md: 2 }}
            spacing={{ base: 10, lg: 32 }}
            py={{ base: 10, sm: 20, lg: 35 }}
        >
            <Hero />
            <Stack
                bg="gray.50"
                rounded="xl"
                p={{ base: 4, sm: 6, md: 8 }}
                spacing={{ base: 8 }}
                maxW={{ lg: 'lg' }}
            >
                <Stack spacing={4}>
                    <Heading
                        color="gray.800"
                        lineHeight={1.1}
                        fontSize={{ base: '2xl', sm: '3xl', md: '4xl' }}
                    >
                        Find the right candidate
                        <Text
                            as="span"
                            bgGradient="linear(to-r, red.400,pink.400)"
                            bgClip="text"
                        >
                            !
                        </Text>
                    </Heading>
                    <Text color="gray.500" fontSize={{ base: 'sm', sm: 'md' }}>
                        This model will help you to look through for the best
                        applicants for a summer internship in Company Co. Please
                        fill the following information and find out if the
                        candidate will fit in our culture.
                    </Text>
                </Stack>
                <Box as="form" mt={10} onSubmit={handleSubmit}>
                    <Grid spacing={6} templateColumns="repeat(2, 1fr)" gap={6}>
                        <FormControl>
                            <FormLabel> Gender </FormLabel>
                            <Select
                                onChange={handleChange}
                                bg="gray.100"
                                border={0}
                                color="gray.500"
                                _placeholder={{
                                    color: 'gray.500',
                                }}
                                name="gender"
                                value={formValues?.gender}
                            >
                                <option hidden>Gender</option>
                                <option value="Male">Male</option>
                                <option value="Female">Female</option>
                            </Select>
                        </FormControl>

                        <FormControl>
                            <FormLabel>Bachelor's Gpa</FormLabel>
                            <Input
                                onChange={handleChange}
                                placeholder="Bachelor's gpa"
                                bg="gray.100"
                                border={0}
                                color="gray.500"
                                type="number"
                                _placeholder={{
                                    color: 'gray.500',
                                }}
                                name="bachelor_score"
                                value={formValues?.bachelor_score}
                            />
                        </FormControl>
                        <FormControl>
                            <FormLabel>Master's Gpa</FormLabel>
                            <Input
                                onChange={handleChange}
                                placeholder="Master's gpa"
                                bg="gray.100"
                                border={0}
                                type="number"
                                color="gray.500"
                                _placeholder={{
                                    color: 'gray.500',
                                }}
                                name="master_score"
                                value={formValues?.master_score}
                            />
                        </FormControl>
                        <FormControl>
                            <FormLabel>Tech test score</FormLabel>
                            <Input
                                onChange={handleChange}
                                placeholder="Test score"
                                bg="gray.100"
                                border={0}
                                type="number"
                                color="gray.500"
                                _placeholder={{
                                    color: 'gray.500',
                                }}
                                name="experience_test"
                                value={formValues?.experience_test}
                            />
                        </FormControl>
                    </Grid>
                    <FormControl mt="1rem">
                        <FormLabel>
                            Applicant has previous experience?{' '}
                        </FormLabel>
                        <Select
                            onChange={handleChange}
                            bg="gray.100"
                            border={0}
                            color="gray.500"
                            _placeholder={{
                                color: 'gray.500',
                            }}
                            name="work_experience"
                            value={formValues?.work_experience}
                        >
                            <option hidden>Previous work experience</option>
                            <option>Yes</option>
                            <option>No</option>
                        </Select>
                    </FormControl>
                    <Button
                        type="submit"
                        fontFamily="heading"
                        mt={8}
                        w="full"
                        bgGradient="linear(to-r, red.400,pink.400)"
                        color="white"
                        _hover={{
                            bgGradient: 'linear(to-r, red.400,pink.400)',
                            boxShadow: 'xl',
                        }}
                    >
                        Submit
                    </Button>
                </Box>
            </Stack>
        </Container>
    );
}
